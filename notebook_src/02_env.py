# ==============================================================================
# 1. ENVIRONMENT, IMPORTS, GPU SETTINGS, HELPERS
# ==============================================================================
if PLATFORM == "colab":
    from google.colab import drive      # already mounted in the config cell (needed here for flush_and_unmount)

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, Markdown

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
_CPU_BF16 = DEVICE.type != "cuda" and os.environ.get("XGUARD_CPU_BF16") == "1"   # developer check only: bf16 autocast on CPU
if DEVICE.type != "cuda" and not _CPU_BF16:
    C.AMP = False
# A100/L4/H100 -> bfloat16 (no loss scaling needed); T4/P100/V100 -> float16 + GradScaler
AMP_DTYPE = torch.bfloat16 if (_CPU_BF16 or (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())) else torch.float16
if _CPU_BF16:   # emulate CUDA: these ops have no bf16/fp16 kernel on the GPU, so make them fail on CPU too
    def _strict(fn):
        def g(x, *a, **k):
            if torch.is_tensor(x) and x.dtype in (torch.bfloat16, torch.float16):
                raise RuntimeError(f"{fn.__name__}: Low precision dtypes not supported. Got {x.dtype}")
            return fn(x, *a, **k)
        return g
    torch.linalg.inv, torch.quantile = _strict(torch.linalg.inv), _strict(torch.quantile)
USE_SCALER = C.AMP and AMP_DTYPE == torch.float16
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| device", DEVICE, GPU_NAME, "| amp", AMP_DTYPE if C.AMP else "off")
if MODE in ("test", "full") and DEVICE.type != "cuda" and not SMOKE:
    raise RuntimeError("No GPU! Colab: Runtime -> Change runtime type -> A100 GPU.  Kaggle: Settings -> Accelerator -> GPU. "
                       "(Your previous run trained on CPU: 1 epoch took 10.5 hours.)")


def release_runtime(reason):
    """Colab: flush Drive and give the GPU back so no compute units are spent on an idle runtime."""
    if PLATFORM != "colab" or not AUTO_DISCONNECT:
        return
    print(f"⏏  releasing the Colab runtime: {reason} (all results are on Drive)")
    try:
        drive.flush_and_unmount()
    except Exception:
        pass
    from google.colab import runtime
    runtime.unassign()


if PLATFORM == "colab" and MODE in ("test", "full") and DEVICE.type == "cuda" and AUTO_DISCONNECT:
    import threading

    def _gpu_watchdog():
        idle = 0
        while True:
            time.sleep(60)
            try:
                u = int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0])
            except Exception:
                continue
            idle = idle + 1 if u < 3 else 0
            if idle >= globals().get("IDLE_LIMIT_NOW", IDLE_DISCONNECT_MIN):   # raised while tables are built on the CPU
                release_runtime(f"GPU idle for {idle} min (finished, crashed or stuck)")
                return
    threading.Thread(target=_gpu_watchdog, daemon=True).start()
    print(f"GPU idle watchdog on: the runtime is released after {IDLE_DISCONNECT_MIN} idle minutes")


def has_internet(timeout=3):
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except Exception:
        return False


INTERNET = os.environ.get("XGUARD_INTERNET") == "1" or (False if SMOKE else has_internet())
print("Internet:", INTERNET)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAVE_COCO = True
except Exception:
    HAVE_COCO = False
    if INTERNET:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=False)
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            HAVE_COCO = True
        except Exception:
            pass
print("pycocotools:", HAVE_COCO)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def elapsed_h():
    return (time.time() - T_SESSION_START) / 3600.0


def time_left_h():
    return C.MAX_SESSION_HOURS - elapsed_h()


def stable_hash(s, mod=10 ** 9):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:12], 16) % mod


def sha(obj, n=16):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def atomic_write(path, writer):
    """write to <path>.tmp, then rename: a full Drive / disconnect never leaves a half-written file behind."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer(tmp); os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise


def drive_free_gb():
    """free space of the Drive account (Colab), None where unknown."""
    try:
        if PLATFORM == "colab" and Path("/content/drive/MyDrive").exists():
            u = shutil.disk_usage("/content/drive/MyDrive")
            if 1e9 < u.total < 1e14 and u.free <= u.total:        # only trust a plausible Drive quota
                return u.free / 1e9
    except Exception:
        pass
    return None


def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def jload(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def environment_info():
    info = dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if DEVICE.type == "cuda" else None,
                gpu=GPU_NAME, amp=str(AMP_DTYPE) if C.AMP else "off", platform=PLATFORM, code_version=CODE_VERSION)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "n/a"
    except Exception:
        info["git_commit"] = "n/a"
    return info


ENV_INFO = environment_info()
jdump(ENV_INFO, WORK / "logs" / "environment.json")
seed_everything(C.SEED)
