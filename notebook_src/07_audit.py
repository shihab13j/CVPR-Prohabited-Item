# ==============================================================================
# 6. DATA-LEAKAGE AUDIT (runs once in "prepare"; the exclusion policy is FROZEN in the pack
#    before any model is evaluated -- test performance never influences exclusions)
#   (1) byte-identical duplicates : SHA-256 of the file bytes
#   (2) visual near-duplicates    : 64-bit dHash candidates (LSH, Hamming <= DUP_HAMMING) CONFIRMED by a
#                                   16x16 grey-thumbnail mean |diff| <= DUP_MAD (X-ray bags share backgrounds)
#   (3) annotation conflicts      : same image content with different label sets
#   Duplicates are merged into clusters (union-find). Any val/test member of a cluster that also has a TRAIN
#   member (any dataset) is excluded from evaluation; cross-dataset clusters are reported separately.
#   Files: audit/exact_duplicate_report.csv, near_duplicate_report.csv, cross_dataset_overlap.csv,
#          excluded_images.csv, final_split_manifest.csv
# ==============================================================================
from multiprocessing import Pool


def image_fingerprint(path):
    """-> (sha256 of bytes, 64-bit dHash, 16x16 grey thumbnail bytes)"""
    try:
        data = open(path, "rb").read()
        h = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            try:
                im.draft("L", (64, 64))
            except Exception:
                pass
            g = im.convert("L")
            a = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
            t = np.asarray(g.resize((16, 16), Image.BILINEAR), dtype=np.uint8).tobytes()
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return h, int("".join("1" if b else "0" for b in bits), 2), t
    except Exception:
        return None, -1, b""


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def run_audit():
    cache = CACHE / "audit_fingerprints.pkl"
    have = pickle.load(open(cache, "rb")) if cache.exists() else {}
    recs = []
    for k, rs in REC.items():
        rs = sorted(rs, key=lambda r: stable_hash(r["uid"]))
        recs += rs[: C.AUDIT_MAX_PER_DS] if C.AUDIT_MAX_PER_DS else rs
    todo = [r["orig_path"] for r in recs if r["orig_path"] not in have]
    if todo:
        t0 = time.time()
        with Pool(max(1, os.cpu_count() or 2)) as pool:
            res = pool.map(image_fingerprint, todo, chunksize=64)
        have.update(zip(todo, res))
        pickle.dump(have, open(cache, "wb"))
        print(f"fingerprinted {len(todo)} images in {(time.time() - t0) / 60:.1f} min")
    F_ = {r["uid"]: have[r["orig_path"]] for r in recs if have[r["orig_path"]][0] is not None}
    uf = UF()
    # (1) exact
    by_sha = defaultdict(list)
    for u, (h, _, _) in F_.items():
        by_sha[h].append(u)
    exact_rows = []
    for h, us in by_sha.items():
        if len(us) > 1:
            labels = {u: tuple(sorted(UID[u]["cls"])) for u in us}
            for u in us[1:]:
                uf.union(us[0], u)
            exact_rows.append(dict(sha256=h, n=len(us), uids="|".join(us), datasets="|".join(sorted({UID[u]["ds"] for u in us})),
                                   splits="|".join(sorted({str(UID[u]["split"]) for u in us})),
                                   annotation_conflict=len(set(labels.values())) > 1))
    # (2) near
    H = {u: v[1] for u, v in F_.items() if v[1] >= 0}
    TH = {u: np.frombuffer(v[2], np.uint8).astype(np.int16) for u, v in F_.items() if v[1] >= 0}
    bands = defaultdict(list)
    for u, h in H.items():
        for b in range(4):
            bands[(b, (h >> (16 * b)) & 0xFFFF)].append(u)
    pairs, n_cand = set(), 0
    for lst in bands.values():
        if 1 < len(lst) < 2000:
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i], lst[j]
                    d = bin(H[a] ^ H[b]).count("1")
                    if d <= C.DUP_HAMMING:
                        n_cand += 1
                        mad = float(np.abs(TH[a] - TH[b]).mean())
                        if mad <= C.DUP_MAD and F_[a][0] != F_[b][0]:
                            pairs.add((min(a, b), max(a, b), d, round(mad, 2)))
    near_rows = []
    for a, b, d, mad in pairs:
        uf.union(a, b)
        near_rows.append(dict(uid_a=a, uid_b=b, dataset_a=UID[a]["ds"], dataset_b=UID[b]["ds"], split_a=UID[a]["split"],
                              split_b=UID[b]["split"], hamming=d, thumb_mad=mad,
                              annotation_conflict=sorted(UID[a]["cls"]) != sorted(UID[b]["cls"])))
    # clusters -> exclusions
    clusters = defaultdict(list)
    for u in list(uf.p):
        clusters[uf.find(u)].append(u)
    excluded, cross_rows = {}, []
    for root, us in clusters.items():
        if len(us) < 2:
            continue
        train_members = [u for u in us if UID[u]["split"] == "train"]
        dss = sorted({UID[u]["ds"] for u in us})
        if len(dss) > 1:
            cross_rows.append(dict(cluster=root, size=len(us), datasets="|".join(dss),
                                   members="|".join(f"{UID[u]['ds']}:{UID[u]['split']}:{UID[u]['img']}" for u in us)))
        if train_members:
            for u in us:
                if UID[u]["split"] in ("val", "test"):
                    excluded[u] = f"duplicate cluster with train image {train_members[0]} ({'cross-dataset' if len(dss) > 1 else 'within-dataset'})"
    pd.DataFrame(exact_rows).to_csv(WORK / "audit" / "exact_duplicate_report.csv", index=False)
    pd.DataFrame(near_rows).to_csv(WORK / "audit" / "near_duplicate_report.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(WORK / "audit" / "cross_dataset_overlap.csv", index=False)
    pd.DataFrame([dict(uid=u, dataset=UID[u]["ds"], split=UID[u]["split"], image=UID[u]["img"], reason=v)
                  for u, v in excluded.items()]).to_csv(WORK / "audit" / "excluded_images.csv", index=False)
    summary = dict(images=len(F_), exact_clusters=len(exact_rows), exact_annotation_conflicts=sum(r["annotation_conflict"] for r in exact_rows),
                   near_candidates=n_cand, near_confirmed=len(near_rows), cross_dataset_clusters=len(cross_rows),
                   excluded_eval_images=len(excluded),
                   excluded_by_dataset=dict(Counter(UID[u]["ds"] for u in excluded)))
    return excluded, summary


if NEED_DISCOVERY:
    EXCLUDED, AUDIT_SUMMARY = run_audit()
for r in ALL_REC:
    r["excluded"] = r["uid"] in EXCLUDED
if NEED_DISCOVERY:
    MANIFEST_HASH = write_manifest(EXCLUDED)
    STATS, CLS_DF = dataset_stats()
else:
    MANIFEST_HASH, STATS = _meta["MANIFEST_HASH"], _meta["STATS"]
print("audit:", json.dumps(AUDIT_SUMMARY, default=str))
print("manifest hash:", MANIFEST_HASH)
display(pd.DataFrame([dict(dataset=k, split_origin=SPLIT_ORIGIN.get(k), **{f"{sp}_img": v["splits"][sp]["images"] for sp in ["train", "val", "test"]},
                           test_subsets=v["splits"]["test"]["subsets"]) for k, v in STATS.items()]))
