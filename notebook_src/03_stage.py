# ==============================================================================
# 2. DATA STAGING
#   Colab "prepare" (CPU runtime): Kaggle API -> /content/raw (local SSD) -> discovery / verification / audit
#                                  -> compact pack on Google Drive (images resized to PACK_LONG, JPEG q95,
#                                  boxes kept in ORIGINAL coordinates, so metrics are unchanged).
#   Colab "test"/"full" (GPU)    : copy + untar the pack from Drive to local SSD (a few minutes, no Kaggle download).
#   Kaggle "prepare" (CPU session): datasets attached as inputs -> same pipeline -> pack in /kaggle/working/pack_v2.
#   Kaggle "test"/"full" (GPU)   : attach the prepare notebook's OUTPUT as input; the pack is found automatically.
# ==============================================================================
PACK_READY = False
PACK_EVENTS = {}


def disk_free_gb(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def _extract(k, shard):
    marker = PACK_LOCAL / f".done_{k}"
    if not marker.exists():
        subprocess.run(["tar", "-xf", str(shard), "-C", str(PACK_LOCAL)], check=True)
        marker.write_text("ok")
    PACK_EVENTS[k].set()


def restore_pack(first=("sixray",)):
    """Copy the pack to local SSD. The source-training dataset (SIXray) is extracted FIRST so training can start
    immediately; the other datasets are extracted by a background thread while the GPU is already busy."""
    if not (PACK_DRIVE / "meta.pkl").exists():
        return (PACK_LOCAL / "meta.pkl").exists()
    import threading
    PACK_LOCAL.mkdir(parents=True, exist_ok=True)
    shutil.copy(PACK_DRIVE / "meta.pkl", PACK_LOCAL / "meta.pkl")
    for extra in ["clip_text.pt", "torch"]:
        src = PACK_DRIVE / extra
        if src.is_dir():
            shutil.copytree(src, PACK_LOCAL / extra, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy(src, PACK_LOCAL / extra)
    shards = {f.stem.replace("images_", ""): f for f in sorted(PACK_DRIVE.glob("images_*.tar"))}
    for k in shards:
        PACK_EVENTS[k] = threading.Event()
    t0 = time.time()
    for k in [k for k in first if k in shards]:
        _extract(k, shards[k])
    print(f"pack {PACK_DRIVE}: {', '.join(first)} ready in {(time.time() - t0) / 60:.1f} min; "
          f"{len(shards) - len(first)} more shards extracting in the background")

    def bg():
        for k, f in shards.items():
            if k not in first:
                _extract(k, f)
    threading.Thread(target=bg, daemon=True).start()
    return True


def wait_pack(ds):
    ev = PACK_EVENTS.get(ds)
    if ev is not None and not ev.is_set():
        t0 = time.time(); print(f"   waiting for the {ds} images to finish extracting…", end=" ")
        ev.wait(); print(f"{time.time() - t0:.0f}s")


def wait_pack_all():
    for k in list(PACK_EVENTS):
        wait_pack(k)


def resume_previous_outputs():
    """Kaggle: copy finished runs from previous versions of this notebook attached as input."""
    if PLATFORM != "kaggle":
        return
    n = 0
    for pat in ["/kaggle/input/*/xguard", "/kaggle/input/*/*/xguard", "/kaggle/input/*/*/*/xguard"]:
        for cand in glob.glob(pat):
            for rd_root in Path(cand).glob("runs*"):
                for rd in rd_root.iterdir():
                    dst = WORK / rd_root.name / rd.name
                    if rd.is_dir() and not dst.exists():
                        shutil.copytree(rd, dst); n += 1
    if n:
        print(f"[resume] copied {n} finished run folders from the previous notebook output")


def _set_kaggle_token(tok):
    """New-style Kaggle API token (KGAT_...): env var + ~/.kaggle/access_token (what the kaggle CLI reads)."""
    tok = tok.strip()
    os.environ["KAGGLE_API_TOKEN"] = tok
    d = Path.home() / ".kaggle"; d.mkdir(parents=True, exist_ok=True)
    (d / "access_token").write_text(tok); os.chmod(d / "access_token", 0o600)


def kaggle_auth():
    """Accepts BOTH Kaggle credential styles, never from the code itself:
       new : Colab secret KAGGLE_API_TOKEN (value starts with KGAT_)
       old : Colab secrets KAGGLE_USERNAME + KAGGLE_KEY, or MyDrive/xguard/kaggle.json"""
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_USERNAME")):
        return "environment"
    secrets = {}
    try:
        from google.colab import userdata
        for name in ["KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"]:
            try:
                v = userdata.get(name)
                if v:
                    secrets[name] = v
            except Exception:
                pass
    except Exception:
        pass
    tok = secrets.get("KAGGLE_API_TOKEN") or (secrets.get("KAGGLE_KEY") if str(secrets.get("KAGGLE_KEY", "")).startswith("KGAT_") else None)
    if tok:
        _set_kaggle_token(tok); return "Colab secret (API token)"
    if secrets.get("KAGGLE_USERNAME") and secrets.get("KAGGLE_KEY"):
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = secrets["KAGGLE_USERNAME"], secrets["KAGGLE_KEY"]
        return "Colab secrets (username + key)"
    for p in [Path(DRIVE_DIR) / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"]:
        if p.exists():
            d = json.load(open(p)); os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = d["username"], d["key"]
            return str(p)
    if (Path.home() / ".kaggle" / "access_token").exists():
        return "~/.kaggle/access_token"
    raise RuntimeError("Kaggle credentials not found. Colab: key icon (Secrets) -> add a secret named KAGGLE_API_TOKEN "
                       "whose value is your token (KGAT_...) from kaggle.com -> Settings -> API -> Create New Token, "
                       "switch 'Notebook access' ON, then Run all again.")


def kaggle_download(key, slug, file=None):
    out = Path(INPUT_ROOT) / key
    if (out / ".done").exists():
        return f"{key}: cached"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = ["kaggle", "datasets", "download"]
    extra = ["-p", str(out)] + (["-f", file] if file else [])
    r = subprocess.run(base + [slug] + extra, capture_output=True, text=True)
    if r.returncode != 0:     # older kaggle CLI needs -d
        r = subprocess.run(base + ["-d", slug] + extra, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kaggle download failed for {slug}: {r.stderr[-500:]}")
    for z in list(out.glob("*.zip")):
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(out)], check=True)
        z.unlink()
    (out / ".done").write_text(slug)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    return f"{key}: {slug}{' / ' + file if file else ''} -> {size:.1f} GB in {(time.time() - t0) / 60:.1f} min"


if MODE in ("test", "full", "report"):
    resume_previous_outputs()
    PACK_READY = restore_pack()
    if not PACK_READY and PLATFORM == "colab":
        raise RuntimeError(f"No data pack at {PACK_DRIVE}. First run this notebook once with MODE='prepare' on a CPU "
                           "runtime (Runtime -> Change runtime type -> CPU); it downloads the Kaggle datasets and builds the pack.")
elif MODE == "prepare":
    if PLATFORM == "colab":
        if (PACK_DRIVE / "meta.pkl").exists():
            print(f"⚠️  a pack already exists at {PACK_DRIVE}. Delete that folder to rebuild it; otherwise switch MODE to 'test'.")
        print(f"free disk: {disk_free_gb('/content'):.0f} GB local, Drive folder {DRIVE_DIR}")
        print("Kaggle credentials:", kaggle_auth())
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=False)   # new tokens need a recent CLI
        jobs = {k: v for k, v in KAGGLE_DATASETS.items() if k != "sixray-p1" or USE_SIXRAY_NEGATIVES}
        from concurrent.futures import ThreadPoolExecutor as _TPE
        par = len(jobs) if disk_free_gb("/content") > 80 else 1   # zip + extracted copy need ~2x space while unzipping
        print(f"downloading {len(jobs)} datasets ({'parallel' if par > 1 else 'sequential: low disk'})")
        with _TPE(par) as ex:                            # network-bound: download datasets in parallel
            for msg in ex.map(lambda kv: kaggle_download(kv[0], *kv[1]), jobs.items()):
                print("  ", msg)
        if os.path.isdir(OPIXRAY_OL_DIR) and glob.glob(os.path.join(OPIXRAY_OL_DIR, "test_knife*.txt")):
            shutil.copytree(OPIXRAY_OL_DIR, Path(INPUT_ROOT) / "opixray-ol", dirs_exist_ok=True)
            print("   OPIXray occlusion lists copied from", OPIXRAY_OL_DIR)
        else:
            print(f"   (optional) OPIXray OL1-3 lists not found in {OPIXRAY_OL_DIR} -> P6 occlusion-level table will be empty")
if MODE == "report":     # merge the other platform's results (zip files dropped into EXTRA_RESULT_DIRS are unpacked)
    for d in EXTRA_RESULT_DIRS:
        for z in Path(d).glob("*.zip") if Path(d).exists() else []:
            out = Path(d) / z.stem
            if not out.exists():
                shutil.unpack_archive(str(z), str(out)); print("unpacked", z)
NEED_DISCOVERY = not PACK_READY
print(f"PACK_READY={PACK_READY}  NEED_DISCOVERY={NEED_DISCOVERY}")
