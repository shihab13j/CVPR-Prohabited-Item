# ==============================================================================
# 5. IMMUTABLE MANIFEST + DATA INTEGRITY
#   * canonical labels, cleaned boxes, subsets (OL1-3, easy/hard/hidden, negative)
#   * LINEAGE: every image is traced to its ORIGINAL image id (Roboflow names such as
#     "P00031_jpg.rf.<hash>.jpg" -> P00031). Augmented copies are collapsed to ONE copy per id and an id
#     may live in only one split (test > val > train), so no original image leaks across splits.
#   * split origin per dataset (official / mirror / derived) + disjointness asserts
#   * verification against the official dataset sizes (dataset_verification.csv)
#   * class_presence_audit.csv (decides which LOCO folds are valid)
# ==============================================================================
OFFICIAL = {   # sizes published by the dataset papers / official repositories
    "pidray":  dict(train=29457, test=18220, easy=9482, hard=3733, hidden=5005),
    "hixray":  dict(train=36295, test=9069),
    "opixray": dict(train=7109, test=1776),
    "sixray":  dict(positives=8929),
}
SPLIT_ORIGIN = {"pidray": "official (COCO json per split)", "hixray": "official (train/ test/ folders)",
                "opixray": "official, reconstructed (mirror val 888 + test 888 = official test 1,776)",
                "sixray": "MIRROR split (Roboflow export khanhbtq99/sixray), NOT an official SIXray detection split",
                "clcxray": "mirror"}


class StopExecution(Exception):
    def _render_traceback_(self):
        return []


def lineage_id(ds, img):
    s = os.path.basename(str(img))
    if ds == "sixray":
        m = re.match(r"^([PN]\d+)", s, re.I)
        if m:
            return m.group(1).upper()
    if ds == "opixray":
        m = re.search(r"(\d{5,})", s)
        if m:
            return m.group(1)
    s = re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", "", os.path.splitext(s)[0], flags=re.I)
    return s


def hash_split(key, test_frac=0.2):
    return "test" if stable_hash(key) % 1000 < test_frac * 1000 else "train"


def collapse_lineage(k, recs, log):
    """One record per original id; an id lives in one split only (priority test > val > train)."""
    prio = {"test": 0, "val": 1, "train": 2, None: 3}
    groups = defaultdict(list)
    for r in recs:
        groups[r["lineage"]].append(r)
    keep, n_cross, n_copies = [], 0, 0
    for lid, rs in groups.items():
        splits = {r["split"] for r in rs}
        if len(splits) > 1:
            n_cross += 1
        best = min(splits, key=lambda s: prio.get(s, 3))
        cand = sorted([r for r in rs if r["split"] == best], key=lambda r: stable_hash(r["path"]))
        n_copies += len(rs) - 1
        keep.append(cand[0])
    log.update(files=len(recs), unique_ids=len(groups), dropped_copies=n_copies, ids_in_several_splits=n_cross,
               max_copies_per_id=max((len(v) for v in groups.values()), default=0))
    return keep


def build_records():
    REC, split_policy, lineage_log = {}, {}, {}
    for k, raw in RAW.items():
        recs, dropped = [], 0
        for r in raw:
            W, H = r.get("w"), r.get("h")
            if not W or not H:
                continue
            bx, cl = [], []
            for (c, x1, y1, x2, y2) in r["objs"]:
                x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
                x1, y1 = max(0.0, x1), max(0.0, y1); x2, y2 = min(float(W), x2), min(float(H), y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    dropped += 1; continue
                bx.append([x1, y1, x2, y2]); cl.append(canon(c))
            recs.append(dict(ds=k, img=r["img"], path=r["path"], orig_path=r["path"], split=r["split"],
                             subsets=list(r["subsets"]), w=int(W), h=int(H), boxes=np.array(bx, np.float32).reshape(-1, 4),
                             cls=cl, neg=len(bx) == 0, lineage=lineage_id(k, r["img"])))
        policy = []
        if k in MERGE_VAL_INTO_TEST and any(r["split"] == "val" for r in recs):
            nv = sum(r["split"] == "val" for r in recs)
            for r in recs:
                if r["split"] == "val":
                    r["split"] = "test"
            policy.append(f"mirror val ({nv}) + test merged back into the official test set")
        if k == "sixray":   # Roboflow images whose label file is empty are NOT clean bags: drop them
            n0 = len(recs); recs = [r for r in recs if not (r["neg"] and r["lineage"].startswith("P"))]
            if n0 - len(recs):
                policy.append(f"{n0 - len(recs)} positive-id images with empty labels dropped")
        lineage_log[k] = {}
        recs = collapse_lineage(k, recs, lineage_log[k])
        if lineage_log[k]["dropped_copies"] or lineage_log[k]["ids_in_several_splits"]:
            policy.append(f"lineage: {lineage_log[k]['files']} files -> {lineage_log[k]['unique_ids']} original ids "
                          f"({lineage_log[k]['dropped_copies']} augmented/duplicate copies dropped, "
                          f"{lineage_log[k]['ids_in_several_splits']} ids appeared in >1 split)")
        splits = Counter(r["split"] for r in recs)
        if splits.get("test", 0) == 0:
            for r in recs:
                r["split"] = hash_split(k + r["lineage"])
            policy.append("no test split shipped -> deterministic md5 split 80/20 by original id (CUSTOM split)")
        else:
            for r in recs:
                if r["split"] is None:
                    r["split"] = hash_split(k + r["lineage"])
            policy.append("train/test split as shipped")
        if not any(r["split"] == "val" for r in recs):
            for r in recs:
                if r["split"] == "train" and stable_hash("val" + k + r["lineage"]) % 10 == 0:
                    r["split"] = "val"
            policy.append("val = 10% of train by original id (thresholds / checkpoint selection ONLY)")
        # clean (negative) bags = SIXray N*.jpg from the official archive part
        negs = sorted([p for p in RAW_NEG.get(k, []) if k != "sixray" or re.match(r"^N\d+", os.path.basename(p), re.I)],
                      key=lambda p: stable_hash(p))
        n_need = C.N_NEG_TRAIN + C.N_NEG_EVAL
        neg_recs = [dict(ds=k, img=os.path.splitext(os.path.basename(p))[0], path=p, orig_path=p, split=None,
                         subsets=["negative"], w=None, h=None, boxes=np.zeros((0, 4), np.float32), cls=[], neg=True,
                         lineage=lineage_id(k, p)) for p in negs[:n_need]]
        if neg_recs:
            read_sizes(neg_recs)
            for i, r in enumerate(neg_recs):
                r["split"] = "train" if i < C.N_NEG_TRAIN else "test"
            neg_recs = [r for r in neg_recs if r["w"]]
            policy.append(f"{len(neg_recs)} clean bags (train {C.N_NEG_TRAIN} / test {C.N_NEG_EVAL})")
        recs += neg_recs
        for r in recs:
            r["uid"] = f"{k}/{hashlib.md5(r['orig_path'].encode()).hexdigest()[:14]}"
            if r["neg"] and "negative" not in r["subsets"]:
                r["subsets"].append("negative")
        REC[k] = recs
        split_policy[k] = policy
        print(f"[{k}] records={len(recs)} dropped-degenerate-boxes={dropped}  " + " | ".join(policy))
    return REC, split_policy, lineage_log


def split_metadata():
    meta, rows = {}, []
    for k, recs in REC.items():
        ids = {sp: {r["lineage"] for r in recs if r["split"] == sp} for sp in ["train", "val", "test"]}
        assert ids["train"].isdisjoint(ids["test"]), f"{k}: train/test share original ids"
        assert ids["val"].isdisjoint(ids["test"]), f"{k}: val/test share original ids"
        assert ids["train"].isdisjoint(ids["val"]), f"{k}: train/val share original ids"
        meta[k] = dict(dataset=k, source=DISCOVERY.get(k, {}).get("roots"), split_origin=SPLIT_ORIGIN.get(k, "as shipped"),
                       policy=SPLIT_POLICY.get(k), lineage=LINEAGE_LOG.get(k),
                       n_ids={sp: len(v) for sp, v in ids.items()}, ids_sha={sp: sha(sorted(v)) for sp, v in ids.items()})
        obs = Counter(r["split"] for r in recs if not r["neg"])
        sub = Counter(s for r in recs if r["split"] == "test" for s in r["subsets"])
        observed = dict(train=obs["train"] + obs["val"], test=obs["test"], positives=sum(obs.values()), **sub)
        for key, off in OFFICIAL.get(k, {}).items():
            got = observed.get(key, 0)
            rows.append(dict(dataset=k, quantity=key, official=off, observed=got, diff=got - off,
                             status="match" if got == off else ("close (<0.1%)" if abs(got - off) <= max(3, 0.001 * off) else "MISMATCH")))
    jdump(meta, WORK / "audit" / "split_metadata.json")
    ver = pd.DataFrame(rows)
    ver.to_csv(WORK / "audit" / "dataset_verification.csv", index=False)
    return meta, ver


def class_presence_audit():
    rows = []
    for k, recs in REC.items():
        classes = sorted({c for r in recs for c in r["cls"]})
        for c in classes:
            cnt = {sp: sum(r["cls"].count(c) for r in recs if r["split"] == sp) for sp in ["train", "val", "test"]}
            img = {sp: sum(c in r["cls"] for r in recs if r["split"] == sp) for sp in ["train", "val", "test"]}
            rows.append(dict(dataset=k, cls=c, superclass=superclass(c), train_objects=cnt["train"], val_objects=cnt["val"],
                             test_objects=cnt["test"], train_images=img["train"], test_images=img["test"],
                             valid_loco=bool(k == "sixray" and cnt["train"] >= C.LOCO_MIN_OBJ and cnt["test"] >= C.LOCO_MIN_OBJ)))
    df = pd.DataFrame(rows)
    df.to_csv(WORK / "audit" / "class_presence_audit.csv", index=False)
    return df


if PACK_READY:
    _meta = pickle.load(open(PACK_LOCAL / "meta.pkl", "rb"))
    REC, DISCOVERY, SPLIT_POLICY, LINEAGE_LOG = _meta["REC"], _meta["DISCOVERY"], _meta["SPLIT_POLICY"], _meta["LINEAGE_LOG"]
    EXCLUDED, AUDIT_SUMMARY = _meta["EXCLUDED"], _meta.get("AUDIT_SUMMARY", {})
    for r in (r for v in REC.values() for r in v):
        r["path"] = str(PACK_LOCAL / r["pack_rel"])
    DATASET_ROOTS_FOUND = {k: DISCOVERY[k].get("roots") for k in REC}
    print(f"loaded pack {_meta['pack_id']} built {_meta['built']}: " + ", ".join(f"{k}={len(v)}" for k, v in REC.items()))
else:
    REC, SPLIT_POLICY, LINEAGE_LOG = build_records()
    EXCLUDED, AUDIT_SUMMARY = {}, {}
ALL_REC = [r for k in REC for r in REC[k]]
UID = {r["uid"]: r for r in ALL_REC}
if PACK_READY:   # audits of the FULL datasets, computed once in the prepare step
    SPLIT_META, VERIFICATION, CLASS_AUDIT = _meta["SPLIT_META"], _meta["VERIFICATION"], _meta["CLASS_AUDIT"]
else:
    SPLIT_META, VERIFICATION = split_metadata()
    CLASS_AUDIT = class_presence_audit()
print("\nVerification against official dataset sizes:")
display(VERIFICATION)
display(CLASS_AUDIT)
display(pd.DataFrame(LINEAGE_LOG).T)


def dataset_stats():
    S, rows = {}, []
    for k, recs in REC.items():
        d = dict(root=DATASET_ROOTS_FOUND.get(k), format=DISCOVERY[k]["format"], split_policy=SPLIT_POLICY[k],
                 split_origin=SPLIT_ORIGIN.get(k), splits={})
        for sp in ["train", "val", "test"]:
            rs = [r for r in recs if r["split"] == sp and not r.get("excluded")]
            cls_cnt = Counter(c for r in rs for c in r["cls"])
            areas = np.array([((b[2] - b[0]) * (b[3] - b[1])) / (r["w"] * r["h"]) for r in rs for b in r["boxes"]] or [np.nan])
            sub = Counter(s for r in rs for s in r["subsets"])
            d["splits"][sp] = dict(images=len(rs), positives=sum(not r["neg"] for r in rs), negatives=sum(r["neg"] for r in rs),
                                   instances=int(sum(cls_cnt.values())), classes=dict(cls_cnt.most_common()),
                                   subsets=dict(sub), rel_box_area_median=float(np.nanmedian(areas)),
                                   img_w_median=float(np.median([r["w"] for r in rs])) if rs else None,
                                   img_h_median=float(np.median([r["h"] for r in rs])) if rs else None)
            for c, n in cls_cnt.items():
                rows.append(dict(dataset=k, split=sp, cls=c, superclass=superclass(c), instances=n))
        S[k] = d
    jdump(S, WORK / "dataset_stats.json")
    return S, pd.DataFrame(rows)


raw_names = defaultdict(set)
for k, d in DISCOVERY.items():
    for n in d.get("raw_class_names", d.get("classes", {})):
        raw_names[k].add(n)
jdump(dict(raw_to_canonical={k: {n: canon(n) for n in sorted(v)} for k, v in raw_names.items()},
           semantic_aliases=SEM_ALIAS if C.USE_SEMANTIC_ALIASES else {}, superclasses=SUPER, attributes=ATTR_GROUPS,
           class_attributes={c: sorted(v) for c, v in CLASS_ATTR.items()}, ontology_version=ONTOLOGY_VERSION),
      WORK / "class_mapping.json")


def write_manifest(excluded=None):
    excluded = excluded or {}
    rows = [dict(uid=r["uid"], dataset=r["ds"], image=r["img"], lineage_id=r["lineage"], source_path=r["orig_path"],
                 split=r["split"], subsets="|".join(r["subsets"]), width=r["w"], height=r["h"], n_boxes=len(r["boxes"]),
                 classes="|".join(sorted(set(r["cls"]))), negative=r["neg"], excluded_reason=excluded.get(r["uid"], ""))
            for r in ALL_REC]
    df = pd.DataFrame(rows)
    df.to_csv(WORK / "audit" / "final_split_manifest.csv", index=False)
    return sha(df[["uid", "split", "excluded_reason"]].values.tolist())
