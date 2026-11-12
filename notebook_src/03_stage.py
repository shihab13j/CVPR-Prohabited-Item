# ==============================================================================
# 2. DATA STAGING
#   Colab "prepare" (CPU runtime): Kaggle API -> /content/raw (local SSD) -> discovery / verification / audit
#                                  -> compact pack on Google Drive (images resized to PACK_LONG, JPEG q95,
#                                  boxes kept in ORIGINAL coordinates, so metrics are unchanged).
#   Colab "test"/"full" (GPU)    : copy + untar the pack from Drive to local SSD (a few minutes, no Kaggle download).
#   Kaggle platform              : datasets are read directly from /kaggle/input (no pack needed).
# ==============================================================================
PACK_READY = False


def disk_free_gb(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def restore_pack():
    if (PACK_LOCAL / "meta.pkl").exists() and (PACK_LOCAL / ".restored").exists():
        return True
    if not (PACK_DRIVE / "meta.pkl").exists():
        return False
    PACK_LOCAL.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shards = sorted(PACK_DRIVE.glob("images_*.tar"))
    print(f"restoring pack from {PACK_DRIVE}: {len(shards)} shards, "
          f"{sum(s.stat().st_size for s in shards) / 1e9:.1f} GB -> {PACK_LOCAL}")
    for s in shards:
        subprocess.run(["tar", "-xf", str(s), "-C", str(PACK_LOCAL)], check=True)
    shutil.copy(PACK_DRIVE / "meta.pkl", PACK_LOCAL / "meta.pkl")
    (PACK_LOCAL / ".restored").write_text("ok")
    print(f"pack restored in {(time.time() - t0) / 60:.1f} min")
    return True


def kaggle_auth():
    if os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_USERNAME"):
        return
    try:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        return
    except Exception:
        pass
    for p in [Path(DRIVE_DIR) / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"]:
        if p.exists():
            d = json.load(open(p)); os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = d["username"], d["key"]
            return
    raise RuntimeError("Kaggle API key not found. Colab: key icon (Secrets) -> add KAGGLE_USERNAME and KAGGLE_KEY "
                       "(from kaggle.com -> Settings -> API -> Create New Token), enable notebook access, re-run.")


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


if MODE in ("test", "full"):
    PACK_READY = restore_pack()
    if not PACK_READY and PLATFORM == "colab":
        raise RuntimeError(f"No data pack at {PACK_DRIVE}. First run this notebook once with MODE='prepare' on a CPU "
                           "runtime (Runtime -> Change runtime type -> CPU); it downloads the Kaggle datasets and builds the pack.")
elif MODE == "prepare":
    if PLATFORM == "colab":
        if (PACK_DRIVE / "meta.pkl").exists():
            print(f"⚠️  a pack already exists at {PACK_DRIVE}. Delete that folder to rebuild it; otherwise switch MODE to 'test'.")
        print(f"free disk: {disk_free_gb('/content'):.0f} GB local, Drive folder {DRIVE_DIR}")
        kaggle_auth()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=False)
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
NEED_DISCOVERY = not PACK_READY
print(f"PACK_READY={PACK_READY}  NEED_DISCOVERY={NEED_DISCOVERY}")
