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
    keep = []
    for k, recs in REC.items():
        tr = sorted([r for r in recs if r["split"] == "train" and not r["neg"]], key=train_cap_order)
        if C.MAX_TRAIN_PER_DS:
            tr = tr[: C.MAX_TRAIN_PER_DS]
        keep += tr + [r for r in recs if r["split"] != "train" or r["neg"]]
    return keep


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
    for k in PACKED_REC:
        subprocess.run(["tar", "-cf", str(PACK_DRIVE / f"images_{k}.tar"), "-C", str(PACK_LOCAL), f"images/{k}"], check=True)
    meta = dict(REC=PACKED_REC, DISCOVERY=DISCOVERY, SPLIT_POLICY=SPLIT_POLICY, LINEAGE_LOG=LINEAGE_LOG, EXCLUDED=EXCLUDED,
                AUDIT_SUMMARY=AUDIT_SUMMARY, STATS=STATS, VERIFICATION=VERIFICATION, CLASS_AUDIT=CLASS_AUDIT,
                SPLIT_META=SPLIT_META, MANIFEST_HASH=MANIFEST_HASH, ONTOLOGY_VERSION=ONTOLOGY_VERSION,
                pack_id=pack_id, built=time.strftime("%Y-%m-%d %H:%M"), PACK_LONG=C.PACK_LONG, PACK_QUALITY=C.PACK_QUALITY)
    pickle.dump(meta, open(PACK_DRIVE / "meta.pkl", "wb"))
    size = sum(f.stat().st_size for f in PACK_DRIVE.glob("images_*.tar")) / 1e9
    print(f"\n✅ pack {pack_id} written to {PACK_DRIVE} ({size:.1f} GB, {len(sel_uids)} images).")
    print("NEXT: Runtime -> Change runtime type -> A100 GPU, set MODE = 'test', Run all.")
    raise StopExecution()
elif PACK_READY:
    print(f"pack {_meta['pack_id']}: verification / class audit / split metadata are those of the FULL datasets (prepare step)")
    VERIFICATION.to_csv(WORK / "audit" / "dataset_verification.csv", index=False)
    CLASS_AUDIT.to_csv(WORK / "audit" / "class_presence_audit.csv", index=False)
    jdump(SPLIT_META, WORK / "audit" / "split_metadata.json")
    jdump(STATS, WORK / "dataset_stats.json")
