# ==============================================================================
# 0. CONFIGURATION  (edit this cell only)
# ==============================================================================
import os, sys, json, time, math, random, re, glob, shutil, pickle, hashlib, warnings, itertools
import contextlib, io, gc, copy, traceback, platform, subprocess, inspect
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
from types import SimpleNamespace
import numpy as np

warnings.filterwarnings("ignore")
T_SESSION_START = time.time()
CODE_VERSION = "xguard-v2.0"          # bump when you change model/training code on purpose (part of every run fingerprint)

# ------------------------------------------------------------------------------
# WHERE / WHAT TO RUN
#   Colab Pro A100 : Tier 1  = core (base vs XGuard, 3 seeds, full test sets) + component ablation
#   Kaggle GPU     : Tier 2  = LOCO, prototype / unknown-generation / SIRA / occlusion ablations, baselines, zero-shot
#                    Tier 3  = in-domain, LODO, encoder, #sources, YOLO / RT-DETR
#
# MODE
#   "prepare" : CPU runtime/session (no GPU cost). Data -> verification -> leakage audit -> compact pack
#               (+ CLIP text embeddings + COCO weights cached inside the pack). Colab: pack on Google Drive.
#               Kaggle: pack in /kaggle/working (attach this notebook's output as input to the GPU sessions).
#   "test"    : GPU. Every stage on real data with a few iterations + measured time / compute-unit estimate.
#   "full"    : GPU. The real experiments of this platform's tiers (resumable, fingerprinted).
#   "report"  : CPU. Merges Colab + Kaggle results -> all tables, figures, release gates (no GPU cost).
# ------------------------------------------------------------------------------
MODE = "test"
TIERS = None           # None = this platform's default (Colab [1], Kaggle [2, 3]); or e.g. [2]
SMOKE = os.environ.get("XGUARD_SMOKE", "0") == "1"   # tiny CPU self-test used by the author; leave it alone
if os.environ.get("XGUARD_MODE"):
    MODE = os.environ["XGUARD_MODE"]
if os.environ.get("XGUARD_TIERS"):
    TIERS = [int(t) for t in os.environ["XGUARD_TIERS"].split(",")]

# ---- platform & paths -----------------------------------------------------------
PLATFORM = "colab" if "google.colab" in sys.modules or os.path.exists("/content/sample_data") else \
           "kaggle" if os.path.exists("/kaggle") else "local"
if os.environ.get("XGUARD_PLATFORM"):
    PLATFORM = os.environ["XGUARD_PLATFORM"]
if TIERS is None:
    TIERS = [1] if PLATFORM in ("colab", "local") else [2, 3]
AUTO_DISCONNECT = True          # Colab: release the GPU runtime when the work is done (results are on Drive)
IDLE_DISCONNECT_MIN = 20        # Colab: release the GPU runtime if it sits idle this long (crash / stuck cell)

DRIVE_DIR = "/content/drive/MyDrive/xguard"      # Colab: pack + all run outputs live here (survive disconnects)


def mount_drive():
    """Mount Google Drive BEFORE anything is written under /content/drive (Colab refuses to mount over files)."""
    if os.path.ismount("/content/drive"):
        return                                     # already mounted (cell re-run)
    if os.path.isdir("/content/drive") and os.listdir("/content/drive"):
        # local leftovers from a run that wrote here before mounting: move them away (they are NOT on Drive)
        bak = f"/content/drive_local_leftover_{int(time.time())}"
        shutil.move("/content/drive", bak)
        print(f"moved local leftovers {bak} (not your Drive files) so Drive can be mounted")
    from google.colab import drive
    drive.mount("/content/drive")


if PLATFORM == "colab":
    mount_drive()


def _find_kaggle_pack():
    for pat in ["/kaggle/input/*/pack_v2", "/kaggle/input/*/*/pack_v2", "/kaggle/input/*/*/*/pack_v2", "/kaggle/input/*/*/*/*/pack_v2"]:
        hits = [h for h in glob.glob(pat) if os.path.exists(os.path.join(h, "meta.pkl"))]
        if hits:
            return Path(hits[0])
    return Path("/kaggle/working/pack_v2")


if PLATFORM == "colab":
    INPUT_ROOT = "/content/raw"                   # raw Kaggle downloads (prepare mode only, local SSD)
    PACK_LOCAL = Path("/content/pack")            # pack copied from Drive to local SSD for fast loading
    PACK_DRIVE = Path(DRIVE_DIR) / "pack_v2"
    WORK = Path(DRIVE_DIR) / "work"
    CACHE = Path("/content/cache")
    PACK_SCOPE = "colab"                          # SIXray complete + val/test of the other datasets (tier 1 needs no more)
elif PLATFORM == "kaggle":
    INPUT_ROOT = "/kaggle/input"
    PACK_LOCAL = Path("/tmp/xguard_pack")         # /tmp: not counted in the 20 GB output limit
    PACK_DRIVE = Path("/kaggle/working/pack_v2") if MODE == "prepare" else _find_kaggle_pack()
    WORK = Path("/kaggle/working/xguard")
    CACHE = Path("/tmp/xguard_cache")
    PACK_SCOPE = "all"                            # tiers 2-3 also train on PIDray / HiXray / OPIXray
else:
    INPUT_ROOT = os.environ.get("XGUARD_INPUT", "./input")
    PACK_LOCAL = Path(os.environ.get("XGUARD_PACK", "./pack"))
    PACK_DRIVE = Path("./pack_drive")
    WORK = Path(os.environ.get("XGUARD_WORK", "./xguard_work"))
    CACHE = WORK / "cache"
    PACK_SCOPE = os.environ.get("XGUARD_PACK_SCOPE", "all")
if os.environ.get("XGUARD_PACK_DRIVE"):
    PACK_DRIVE = Path(os.environ["XGUARD_PACK_DRIVE"])
if os.environ.get("XGUARD_INPUT"):
    INPUT_ROOT = os.environ["XGUARD_INPUT"]
if os.environ.get("XGUARD_WORK"):
    WORK = Path(os.environ["XGUARD_WORK"])
if os.environ.get("XGUARD_PACK"):
    PACK_LOCAL = Path(os.environ["XGUARD_PACK"])
# Report mode: run folders of the OTHER platform. Colab: unzip the Kaggle "xguard_results_full.zip" into
# MyDrive/xguard/kaggle_results/ (done automatically if the zip is placed there). Kaggle: attach the Colab results.
EXTRA_RESULT_DIRS = [Path(DRIVE_DIR) / "kaggle_results"] if PLATFORM == "colab" else \
    [Path(p) for p in glob.glob("/kaggle/input/*/colab_results")] if PLATFORM == "kaggle" else \
    [Path(p) for p in os.environ.get("XGUARD_EXTRA_RESULTS", "").split(",") if p]
CACHE = Path(os.environ.get("XGUARD_CACHE", str(CACHE)))

# ---- datasets (Kaggle slugs used by the Colab "prepare" step) ---------------------
#   key -> (kaggle slug, single file to download or None for the whole dataset)
KAGGLE_DATASETS = {
    "sixray":    ("khanhbtq99/sixray", None),                         # Roboflow export of SIXray positives (boxes)
    "sixray-p1": ("witsarutsarai/sixray-p1", "JPEGImage.tar.gz00"),   # ONLY the first part: clean bags (negatives)
    "pidray":    ("banan41k/xray-contraband-detection", None),       # PIDray (COCO json, official easy/hard/hidden)
    "hixray":    ("newashish/hixray", None),
    "opixray":   ("alirzaei/opixray-yolo", None),
}
OPIXRAY_OL_DIR = str(Path(DRIVE_DIR) / "opixray_ol")   # put test_knife{,-1,-2,-3}.txt here (optional, enables P6)
USE_SIXRAY_NEGATIVES = True     # download 1 part of SIXRAY_P1 (~4 GB) to get clean bags for false-alarm metrics

# Kaggle-platform overrides (ignored on Colab): absolute dataset roots, None = auto-discovery by name/content
DATASET_ROOTS = {"sixray": None, "opixray": None, "hixray": None, "pidray": None, "clcxray": None}
FORCE_FORMAT = {"sixray": None, "opixray": None, "hixray": None, "pidray": None, "clcxray": None}
RESUME_FROM_INPUT = True
EXTRACT_ARCHIVES = False
SIXRAY_TAR_EXTRACT = "auto"
MERGE_VAL_INTO_TEST = {"opixray"}   # alirzaei/opixray-yolo cut the OFFICIAL test set (1,776) into val 888 + test 888

# ---- training / evaluation settings -----------------------------------------------
C = SimpleNamespace(
    IMG=800, EPOCHS_MAIN=12, EPOCHS_ABL=8, BATCH=8, ACCUM=1, LR=0.01, WARMUP_ITERS=500, WEIGHT_DECAY=1e-4,
    MAX_TRAIN_PER_DS=12000, MAX_VAL=600, MAX_EVAL_MAIN=None, MAX_EVAL_ABL=3000, N_NEG_TRAIN=500, N_NEG_EVAL=500,
    SEEDS_MAIN=[0, 1, 2], SEEDS_ABL=[0], WORKERS=min(12, os.cpu_count() or 2),
    INIT="coco", TRAINABLE_LAYERS=3, EMB_DIM=256, AMP=True, GRAD_CLIP=10.0,
    SCORE_THR=0.01, DETS_PER_IMG=100, UNK_DETS_PER_IMG=50, BOOTSTRAP=1000,
    USE_SEMANTIC_ALIASES=True, DUP_HAMMING=4, DUP_MAD=3.0, AUDIT_MAX_PER_DS=None, KEEP_FULL_CKPT=False,
    LOSS_W=dict(sira=0.5, occ=0.5, rel=0.2, proto=0.2, unknown=0.5, ent=0.1, cons=0.5, risk=0.5, rec=0.1, threat=0.1),
    VIEW2_DET=True,          # detection loss on view 2 only if it is NOT occluded (occluded boxes are noisy labels)
    OCC_P=0.5, OCC_FRAC=(0.15, 0.5),
    CLIP_HF_NAME="openai/clip-vit-base-patch32", REQUIRE_TEXT=None,   # None -> required in "full" mode
    MAX_NAN_FRAC=0.02, VAL_EVERY=1, PACK_LONG=800, PACK_QUALITY=95,
    REFERRAL_BUDGETS=(0.05, 0.10), OCC_SWEEP=(0.0, 0.1, 0.3, 0.5, 0.7), LOCO_MIN_OBJ=30,
    EXTERNAL_RESULTS=None, SEED=0, MAX_SESSION_HOURS=23.5 if PLATFORM == "colab" else 11.4,
    TEST_ITERS=None, TEST_EVAL=None, TEST_VAL=None,
    CU_PER_HOUR={"A100": 11.8, "L4": 4.8, "T4": 1.8, "V100": 4.9},   # Colab compute units/hour (verify on your Resources page)
)
if PLATFORM == "kaggle":      # T4 / P100 have 16 GB: same EFFECTIVE batch (8) as Colab via accumulation -> comparable runs
    C.BATCH, C.ACCUM = 4, 2
if MODE == "test":
    C.TEST_ITERS, C.TEST_EVAL, C.TEST_VAL, C.BOOTSTRAP = 30, 120, 60, 100
    C.SEEDS_MAIN, C.SEEDS_ABL = [0], [0]
if C.REQUIRE_TEXT is None:
    C.REQUIRE_TEXT = MODE == "full"
if SMOKE:   # tiny CPU self-test
    C.__dict__.update(IMG=160, BATCH=2, MAX_TRAIN_PER_DS=24, MAX_VAL=8, N_NEG_TRAIN=2, N_NEG_EVAL=4, WORKERS=0,
                      INIT="none", WARMUP_ITERS=2, BOOTSTRAP=30, TEST_ITERS=3, TEST_EVAL=10, TEST_VAL=8, PACK_LONG=200,
                      LOCO_MIN_OBJ=1, MAX_EVAL_ABL=10, REQUIRE_TEXT=False)

for sub in ["runs", "tables", "figures", "audit", "logs"]:
    (WORK / sub).mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str((PACK_LOCAL if MODE != "prepare" else PACK_DRIVE) / "torch")   # weights travel inside the pack
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
jdump_cfg = {k: v for k, v in vars(C).items()}
json.dump(dict(MODE=MODE, TIERS=TIERS, PLATFORM=PLATFORM, CODE_VERSION=CODE_VERSION, **jdump_cfg),
          open(WORK / f"config_{MODE}.json", "w"), indent=1, default=str)
print(f"MODE={MODE}  TIERS={TIERS}  PLATFORM={PLATFORM}  WORK={WORK}  PACK={PACK_LOCAL}")
