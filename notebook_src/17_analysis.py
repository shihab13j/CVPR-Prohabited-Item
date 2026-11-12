# ==============================================================================
# 16. REPRESENTATION EVIDENCE (SIRA validation)
#   domain probe ↓ (dataset identity from GT-object embeddings), threat probe ↑ (superclass), scanner-
#   counterfactual consistency ↑ (cos(z(x), z(T_phi x))), feature geometry D_same / D_diff, together with
#   source AP and target AP -- a lower domain probe alone is NOT evidence; threat information must survive.
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_emb(tag):
    p = RUNS_DIR / f"{tag}_s0" / "emb.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def probe(X, y, seed=0):
    y = np.asarray(y); cnt = Counter(y)
    keep_cls = [c for c, n in cnt.items() if n >= 5]
    if len(keep_cls) < 2:
        return np.nan
    m = np.isin(y, keep_cls); X, y = X[m], y[m]
    n = min(Counter(y).values()); rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in keep_cls])
    k = min(5, n)
    return float(cross_val_score(LogisticRegression(max_iter=2000), X[idx], y[idx], cv=StratifiedKFold(k, shuffle=True, random_state=seed)).mean())


def geometry(Z, D, Y):
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-6)
    shared = [c for c in set(Y) if len(set(D[Y == c])) >= 2]
    same, diff, rng = [], [], np.random.RandomState(0)
    for c in shared:
        idx = np.where(Y == c)[0]
        for _ in range(min(2000, len(idx) ** 2)):
            i, j = rng.choice(idx, 2)
            if D[i] != D[j]:
                same.append(1 - Zn[i] @ Zn[j])
    for _ in range(4000):
        i, j = rng.randint(0, len(Y), 2)
        if Y[i] != Y[j]:
            diff.append(1 - Zn[i] @ Zn[j])
    return (np.mean(same) if same else np.nan), (np.mean(diff) if diff else np.nan), shared


REP = {}
for tag, lab in [("six_base", "base"), ("scan-rgb", "+ RGB jitter"), ("scan-fourier", "+ Fourier (FDA)"),
                 ("six_sira", "+ SIRA"), ("six_xguard", "XGuard (full)")]:
    e = load_emb(tag)
    if e is None:
        continue
    Z, D, Y = e["z"], e["ds"], e["cls"]
    ds_, dd_, shared = geometry(Z, D, Y)
    REP[lab] = {"domain-probe acc ↓": probe(Z, D), "threat-probe acc ↑": probe(Z, np.array([superclass(c) for c in Y])),
                "scanner-cf cos ↑": float(np.mean(e["scanner_cos"])) if "scanner_cos" in e and len(e["scanner_cos"]) else np.nan,
                "D_same/D_diff ↓": ds_ / dd_ if dd_ else np.nan,
                "source AP50 (SIXray)": val(tag, "sixray", "AP50"), "target AP50 (PIDray)": val(tag, "pidray", "AP50"),
                "n objects": len(Z)}
if REP:
    save_table(pd.DataFrame(REP).T, "T14_sira_validation",
               "SIRA validation: less scanner information AND preserved threat information AND target gains (all required).")
else:
    print("no embeddings yet (analyze=True runs not finished)")
