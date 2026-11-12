# ==============================================================================
# 1. ENVIRONMENT, IMPORTS, GPU SETTINGS, HELPERS
# ==============================================================================
if PLATFORM == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    for sub in ["runs", "tables", "figures", "audit", "logs"]:
        (WORK / sub).mkdir(parents=True, exist_ok=True)

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
if DEVICE.type != "cuda":
    C.AMP = False
# A100/L4/H100 -> bfloat16 (no loss scaling needed); T4/P100/V100 -> float16 + GradScaler
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
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


def has_internet(timeout=3):
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except Exception:
        return False


INTERNET = False if SMOKE else has_internet()
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
