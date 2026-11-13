# ==============================================================================
# 7. PACK (prepare mode): one compact copy of every image the experiments can use
#    * long side resized to C.PACK_LONG (= training resolution), JPEG quality C.PACK_QUALITY
#    * boxes, widths and heights stay in ORIGINAL pixel coordinates -> metrics (incl. AP-S/M/L) unchanged
#    * train images capped exactly like the experiment registry (C.MAX_TRAIN_PER_DS, same md5 order);
#      all val / test / clean-bag images are kept
#    * written as one tar per dataset to Google Drive + meta.pkl (records, verification, audit, policy)
#    GPU sessions only copy these shards to local SSD: no Kaggle download, no image decoding of huge PNGs.
# ==============================================================================
def train_cap_order(r):
    return stable_hash("tr" + r["uid"])


def select_for_pack():
    """PACK_SCOPE 'colab': tier 1 trains only on SIXray -> other datasets ship only val/test (smaller, faster restore).
    PACK_SCOPE 'all'  : Kaggle tiers 2-3 also train on PIDray / HiXray / OPIXray."""
    keep = []
    for k, recs in REC.items():
        tr = sorted([r for r in recs if r["split"] == "train" and not r["neg"]], key=train_cap_order)
        if C.MAX_TRAIN_PER_DS:
            tr = tr[: C.MAX_TRAIN_PER_DS]
        if PACK_SCOPE == "colab" and k != "sixray":
            tr = []
        keep += tr + [r for r in recs if r["split"] != "train" or r["neg"]]
    return keep


def precompute_text_and_weights():
    """Done on the CPU so that GPU sessions never download or load anything except the detector weights."""
    os.environ["TORCH_HOME"] = str(PACK_DRIVE / "torch")
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models import ResNet50_Weights
        for w in [FasterRCNN_ResNet50_FPN_Weights.COCO_V1, ResNet50_Weights.IMAGENET1K_V1]:
            torch.hub.load_state_dict_from_url(w.url, progress=False)
        print("   COCO + ImageNet weights cached in the pack")
    except Exception as e:
        print("   ⚠️  could not cache detector weights (GPU session will download them):", str(e)[:120])
    try:
        import transformers
        from transformers import CLIPModel, CLIPTokenizer
        prompts = all_text_prompts()
        tok, m = CLIPTokenizer.from_pretrained(C.CLIP_HF_NAME), CLIPModel.from_pretrained(C.CLIP_HF_NAME).eval()
        cache = dict(zip(prompts, clip_encode_text(m, tok, prompts)))
        cache["__meta__"] = dict(model=C.CLIP_HF_NAME, transformers=transformers.__version__, n=len(prompts))
        torch.save(cache, PACK_DRIVE / "clip_text.pt")
        print(f"   CLIP text embeddings for {len(prompts)} prompts cached in the pack")
    except Exception as e:
        print("   ⚠️  CLIP text embeddings NOT cached:", str(e)[:150])


def pack_one(args):
    src, dst, long_side, q = args
    try:
        if os.path.exists(dst):
            return True
        with Image.open(src) as im:
            W, H = im.size
            s = min(1.0, long_side / max(W, H))
            try:
                im.draft("RGB", (int(W * s) + 1, int(H * s) + 1))
            except Exception:
                pass
            im = im.convert("RGB")
            if s < 1.0:
                im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR, reducing_gap=2.0)
            im.save(dst, "JPEG", quality=q, optimize=False)
        return True
    except Exception as e:
        return f"{src}: {e}"


if MODE == "prepare":
    sel = select_for_pack()
    pack_id = sha(dict(manifest=MANIFEST_HASH, long=C.PACK_LONG, q=C.PACK_QUALITY, cap=C.MAX_TRAIN_PER_DS, n=len(sel)), 10)
    jobs = []
    for r in sel:
        rel = f"images/{r['ds']}/{r['uid'].split('/')[1]}.jpg"
        r["pack_rel"] = rel
        (PACK_LOCAL / "images" / r["ds"]).mkdir(parents=True, exist_ok=True)
        jobs.append((r["orig_path"], str(PACK_LOCAL / rel), C.PACK_LONG, C.PACK_QUALITY))
    t0 = time.time()
    with Pool(max(1, os.cpu_count() or 2)) as pool:
        res = pool.map(pack_one, jobs, chunksize=32)
    bad = [x for x in res if x is not True]
    print(f"packed {len(jobs) - len(bad)} images in {(time.time() - t0) / 60:.1f} min ({len(bad)} failed)")
    if bad:
        print("first failures:", bad[:5])
    sel_uids = {r["uid"] for r in sel if r.get("pack_rel")}
    PACKED_REC = {k: [r for r in v if r["uid"] in sel_uids] for k, v in REC.items()}
    PACK_DRIVE.mkdir(parents=True, exist_ok=True)
    if not SMOKE:
        import torch, torch.nn.functional as F
        precompute_text_and_weights()
    for k in PACKED_REC:
        subprocess.run(["tar", "-cf", str(PACK_DRIVE / f"images_{k}.tar"), "-C", str(PACK_LOCAL), f"images/{k}"], check=True)
    meta = dict(REC=PACKED_REC, DISCOVERY=DISCOVERY, SPLIT_POLICY=SPLIT_POLICY, LINEAGE_LOG=LINEAGE_LOG, EXCLUDED=EXCLUDED,
                AUDIT_SUMMARY=AUDIT_SUMMARY, STATS=STATS, VERIFICATION=VERIFICATION, CLASS_AUDIT=CLASS_AUDIT,
                SPLIT_META=SPLIT_META, MANIFEST_HASH=MANIFEST_HASH, ONTOLOGY_VERSION=ONTOLOGY_VERSION,
                pack_id=pack_id, built=time.strftime("%Y-%m-%d %H:%M"), PACK_LONG=C.PACK_LONG, PACK_QUALITY=C.PACK_QUALITY,
                PACK_SCOPE=PACK_SCOPE, PLATFORM=PLATFORM)
    pickle.dump(meta, open(PACK_DRIVE / "meta.pkl", "wb"))
    size = sum(f.stat().st_size for f in PACK_DRIVE.glob("images_*.tar")) / 1e9
    print(f"\n✅ pack {pack_id} written to {PACK_DRIVE} ({size:.1f} GB, {len(sel_uids)} images).")
    if PLATFORM == "kaggle":
        print("NEXT: Save Version (Save & Run All) -> new GPU notebook -> Add Input: THIS notebook's output -> MODE='test'.")
    else:
        print("NEXT: Runtime -> Change runtime type -> A100 GPU, set MODE = 'test', Run all.")
    raise StopExecution()
elif PACK_READY:
    print(f"pack {_meta['pack_id']}: verification / class audit / split metadata are those of the FULL datasets (prepare step)")
    VERIFICATION.to_csv(WORK / "audit" / "dataset_verification.csv", index=False)
    CLASS_AUDIT.to_csv(WORK / "audit" / "class_presence_audit.csv", index=False)
    jdump(SPLIT_META, WORK / "audit" / "split_metadata.json")
    jdump(STATS, WORK / "dataset_stats.json")
