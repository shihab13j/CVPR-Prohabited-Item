# ==============================================================================
# 11. EVALUATION
#   closed-set (3 tracks, never mixed): HARMONIZED AP (primary: AP, AP50, AP75, AR100, APs/m/l),
#                                       EXACT AP (xAP, xAP50), HIERARCHICAL superclass AP (hAP, hAP50)
#   open-world : U_AP, U_AP50, U_R50 (one-to-one matching, top-50 unknown dets), U_R50_op (at tau_U),
#                AOSE, WI (at the source-val operating point)
#                locAUROC / locFPR95: known-vs-unknown separability of U for LOCALIZED objects only
#                (reported with locCoverage = fraction of GT objects that were localized at all)
#   reliability: ECE / Brier / NLL; AURC + E-AURC for risk scores {msp, entropy, energy, U, rcr}
#   operational: bag-level referral at fixed budgets (thresholds from SOURCE val): referral rate,
#                missed threats after referral, false alarms per bag after referral
#   every block reports metric_status = ok | not_applicable:<reason> | evaluator_failure:<error>
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
from sklearn.metrics import roc_auc_score, roc_curve

RISK_TYPES = ["msp", "entropy", "energy", "U", "rcr"]


def np_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None] - inter, 1e-6)


def greedy_match(db, ds, gb, thr=0.5):
    """one-to-one greedy matching (highest score first). returns tp flags per det and matched gt index."""
    n = len(db); tp = np.zeros(n, bool); gi = -np.ones(n, int)
    if n == 0 or len(gb) == 0:
        return tp, gi
    order = np.argsort(-ds, kind="mergesort"); iou = np_iou(db[order], gb); used = np.zeros(len(gb), bool)
    for r, k in enumerate(order):
        c = np.where(used, -1, iou[r]); j = int(c.argmax())
        if c[j] >= thr:
            tp[k], gi[k], used[j] = True, j, True
    return tp, gi


def ap_from(scores, tp, n_gt, w=None):
    """COCO-style 101-point interpolated AP; optional per-detection weights (bootstrap)."""
    if n_gt <= 0:
        return np.nan
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="mergesort"); tp = tp[o].astype(np.float64)
    w = np.ones_like(tp) if w is None else w[o]
    ctp, cfp = np.cumsum(w * tp), np.cumsum(w * (1 - tp))
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101); idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0 for i in idx]))


COCO_KEYS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def coco_eval(gts, dets, cats):
    """gts: {img: [(box, cat)]}, dets: [(img, box, score, cat)] -> (stats, per-class AP50)."""
    if not cats or sum(len(v) for v in gts.values()) == 0:
        return {k: np.nan for k in COCO_KEYS}, {}
    cid = {c: i + 1 for i, c in enumerate(cats)}
    if HAVE_COCO:
        images, anns = [], []
        for im, g in gts.items():
            images.append(dict(id=im, width=1, height=1))
            for b, c in g:
                if c in cid:
                    anns.append(dict(id=len(anns) + 1, image_id=im, category_id=cid[c],
                                     bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                                     area=float((b[2] - b[0]) * (b[3] - b[1])), iscrowd=0))
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(); gt.dataset = dict(images=images, annotations=anns, categories=[dict(id=v, name=k) for k, v in cid.items()]); gt.createIndex()
            dl = [dict(image_id=im, category_id=cid[c], bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], score=float(s))
                  for im, b, s, c in dets if c in cid and im in gts]
            if not dl:
                return {k: 0.0 for k in COCO_KEYS}, {c: 0.0 for c in cats}
            dt = gt.loadRes(dl)
            E = COCOeval(gt, dt, "bbox"); E.params.imgIds = list(gts.keys()); E.evaluate(); E.accumulate(); E.summarize()
        stats = {k: float(v) if v >= 0 else np.nan for k, v in zip(COCO_KEYS, E.stats)}
        pc = {}
        for c, i in cid.items():
            p = E.eval["precision"][0, :, i - 1, 0, 2]; p = p[p > -1]
            pc[c] = float(p.mean()) if len(p) else np.nan
        return stats, pc
    res = {k: np.nan for k in COCO_KEYS}; pc = {}; aps = defaultdict(list)
    for c in cats:
        for t in np.arange(0.5, 0.96, 0.05):
            S, TP, ng = [], [], 0
            for im, g in gts.items():
                gb = np.array([b for b, cc in g if cc == c]).reshape(-1, 4); ng += len(gb)
                d = [(b, s) for i2, b, s, cc in dets if i2 == im and cc == c]
                if d:
                    db = np.array([x[0] for x in d]); dsc = np.array([x[1] for x in d])
                    tp, _ = greedy_match(db, dsc, gb, t); S += list(dsc); TP += list(tp)
            aps[round(t, 2)].append(ap_from(np.array(S), np.array(TP, bool), ng))
        pc[c] = aps[0.5][-1]
    res["AP"] = float(np.nanmean([np.nanmean(v) for v in aps.values()])); res["AP50"] = float(np.nanmean(aps[0.5])); res["AP75"] = float(np.nanmean(aps[0.75]))
    return res, pc


def ece_score(conf, correct, bins=15):
    if len(conf) == 0:
        return np.nan
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def risk_coverage(conf, correct):
    """detection-level selective risk; returns (AURC, E-AURC, curve on a 100-point coverage grid)."""
    if len(conf) == 0:
        return np.nan, np.nan, None
    o = np.argsort(-conf, kind="mergesort"); err = np.cumsum(~correct[o]) / np.arange(1, len(o) + 1)
    cov = np.arange(1, len(o) + 1) / len(o)
    aurc = float(err.mean())
    srt = np.sort(~correct)                                        # perfect ranking: all correct detections first
    opt = float((np.cumsum(srt) / np.arange(1, len(srt) + 1)).mean())   # exact (discrete) optimal AURC
    curve = np.interp(np.linspace(0.01, 1, 100), cov, err)
    return aurc, float(aurc - opt), curve


def det_risk(p, kind):
    """per-known-detection risk (higher = more likely wrong)."""
    if kind == "msp":
        return 1 - p["scores"]
    if kind == "entropy":
        return p.get("H", np.full(len(p["scores"]), np.nan))
    if kind == "energy":
        return p.get("E", np.full(len(p["scores"]), np.nan))
    if kind == "U":
        return p["U"]
    return p["R"]


def build_eval_items(preds, recs, known, subset=None):
    items = []
    for i, r in enumerate(recs):
        if r.get("excluded") or preds[i] is None:
            continue
        if subset and subset != "all" and subset not in r["subsets"]:
            continue
        kb, kc, xb, xc, ub = [], [], [], [], []
        for b, c in zip(r["boxes"], r["cls"]):
            hk = resolve_known(c, known, "harmonized")
            if hk is None:
                ub.append(b)
            else:
                kb.append(b); kc.append(hk)
            if c in known:
                xb.append(b); xc.append(c)
        p = preds[i]
        names = np.array([known[l] for l in p["labels"]], dtype=object)
        tp = np.zeros(len(names), bool)
        kbn, kcn = np.array(kb, np.float32).reshape(-1, 4), np.array(kc, dtype=object)
        for c in set(names.tolist()):
            di, gi = np.where(names == c)[0], np.where(kcn == c)[0]
            t, _ = greedy_match(p["boxes"][di], p["scores"][di], kbn[gi], 0.5); tp[di] = t
        ubn = np.array(ub, np.float32).reshape(-1, 4)
        hit_unk = np_iou(p["boxes"], ubn).max(1) >= 0.5 if len(ubn) and len(names) else np.zeros(len(names), bool)
        items.append(dict(i=i, r=r, p=p, names=names, tp=tp, hit_unk=hit_unk, kb=kbn, kc=kcn,
                          xb=np.array(xb, np.float32).reshape(-1, 4), xc=np.array(xc, dtype=object), ub=ubn,
                          all_cls=list(r["cls"]), all_b=r["boxes"], neg=bool(r["neg"])))
    return items


def _block(M, name, fn):
    try:
        reason = fn()
        M["metric_status"][name] = "ok" if reason is None else f"not_applicable:{reason}"
    except Exception as e:
        M["metric_status"][name] = f"evaluator_failure:{type(e).__name__}: {str(e)[:120]}"


def bag_risk(p, kind):
    m = p["scores"] >= 0.05
    if not m.any():
        return 0.0
    v = det_risk(p, kind)[m]
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else np.nan


def evaluate(preds, recs, known, thr, subset=None, fps=None, light=False):
    items = build_eval_items(preds, recs, known, subset)
    M = dict(n_img=len(items), n_known_gt=int(sum(len(it["kb"]) for it in items)),
             n_exact_gt=int(sum(len(it["xb"]) for it in items)), n_unk_gt=int(sum(len(it["ub"]) for it in items)),
             n_neg_img=int(sum(it["neg"] for it in items)), metric_status={})
    curves = {}
    if not items:
        M["metric_status"]["all"] = "not_applicable:no_images"
        return M, curves
    evcats = order_classes({c for it in items for c in it["kc"]})
    M["eval_classes"] = evcats
    dets = [(it["i"], b, s, c) for it in items for b, s, c in zip(it["p"]["boxes"], it["p"]["scores"], it["names"])]

    def closed():
        if M["n_known_gt"] == 0:
            for k in COCO_KEYS:
                M[k] = np.nan
            return "no_shared_known_gt"
        st, pc = coco_eval({it["i"]: list(zip(it["kb"], it["kc"])) for it in items}, dets, evcats)
        M.update(st); M["per_class_AP50"] = pc
    _block(M, "closed_harmonized", closed)
    if light:
        return M, curves

    def exact():
        xc = order_classes({c for it in items for c in it["xc"]})
        if not xc:
            M["xAP"] = M["xAP50"] = np.nan
            return "no_exact_shared_gt"
        st, _ = coco_eval({it["i"]: list(zip(it["xb"], it["xc"])) for it in items}, dets, xc)
        M["xAP"], M["xAP50"] = st["AP"], st["AP50"]
    _block(M, "closed_exact", exact)

    def hier():
        hg = {it["i"]: [(b, superclass(c)) for b, c in zip(it["all_b"], it["all_cls"])] for it in items}
        hcats = sorted({superclass(c) for c in known} & {c for v in hg.values() for _, c in v} - {"other"})
        M["hier_classes"] = hcats
        if not hcats:
            M["hAP"] = M["hAP50"] = np.nan
            return "no_shared_superclass"
        st, pc = coco_eval(hg, [(i, b, s, superclass(c)) for i, b, s, c in dets], hcats)
        M["hAP"], M["hAP50"], M["per_super_AP50"] = st["AP"], st["AP50"], pc
    _block(M, "closed_hierarchical", hier)

    def unknown():
        if M["n_unk_gt"] == 0:
            for k in ["U_AP", "U_AP50", "U_R50", "U_R50_op"]:
                M[k] = np.nan
            return "no_unknown_gt"
        st, _ = coco_eval({it["i"]: [(b, "unknown") for b in it["ub"]] for it in items},
                          [(it["i"], b, s, "unknown") for it in items for b, s in zip(it["p"]["ub"], it["p"]["us"])], ["unknown"])
        M["U_AP"], M["U_AP50"] = st["AP"], st["AP50"]
        hits = hits_op = 0
        for it in items:
            if len(it["ub"]) and len(it["p"]["ub"]):
                t, _ = greedy_match(it["p"]["ub"], it["p"]["us"], it["ub"], 0.5); hits += int(t.sum())
                m = it["p"]["us"] >= thr["tau_U"]
                if m.any():
                    t, _ = greedy_match(it["p"]["ub"][m], it["p"]["us"][m], it["ub"], 0.5); hits_op += int(t.sum())
        M["U_R50"], M["U_R50_op"] = hits / M["n_unk_gt"], hits_op / M["n_unk_gt"]
    _block(M, "unknown", unknown)

    def loc_auroc():
        ys, us, n_gt = [], [], 0
        for it in items:
            B = np.concatenate([it["p"]["boxes"], it["p"]["ub"]]); Uv = np.concatenate([it["p"]["U"], it["p"]["us"]])
            for gb, lab in [(it["kb"], 0), (it["ub"], 1)]:
                n_gt += len(gb)
                if len(gb) and len(B):
                    for row in np_iou(gb, B):
                        if row.max() >= 0.5:
                            ys.append(lab); us.append(float(Uv[row.argmax()]))
        M["locCoverage"] = len(ys) / max(1, n_gt)
        ys, us = np.array(ys), np.array(us)
        if len(set(ys.tolist())) < 2:
            M["locAUROC"] = M["locFPR95"] = M["locAUROC_flipped"] = np.nan
            return "needs_localized_known_and_unknown"
        assert np.isfinite(us).all()
        M["locAUROC"] = float(roc_auc_score(ys, us)); M["locAUROC_flipped"] = float(roc_auc_score(ys, -us))  # diagnostic only
        fpr, tpr, _ = roc_curve(ys, us); M["locFPR95"] = float(fpr[min(len(fpr) - 1, np.searchsorted(tpr, 0.95))])
    _block(M, "loc_auroc", loc_auroc)

    tk, tu, tr = thr["tau_known"], thr["tau_U"], thr["tau_R"]
    S = np.concatenate([it["p"]["scores"] for it in items]); TP = np.concatenate([it["tp"] for it in items])
    NM = np.concatenate([it["names"] for it in items]); Uall = np.concatenate([it["p"]["U"] for it in items])
    HU = np.concatenate([it["hit_unk"] for it in items])

    def openset_op():
        if M["n_unk_gt"] == 0:
            M["AOSE"] = M["WI"] = np.nan
            return "no_unknown_gt"
        sel = S >= tk
        M["AOSE"] = int((sel & ~TP & HU & (Uall < tu)).sum())
        nk_only = np.concatenate([np.full(len(it["p"]["scores"]), len(it["ub"]) == 0) for it in items])
        P_KU = TP[sel].mean() if sel.any() else np.nan
        P_K = TP[sel & nk_only].mean() if (sel & nk_only).any() else np.nan
        M["WI"] = float(P_K / P_KU - 1) if P_KU and np.isfinite(P_KU) and np.isfinite(P_K) else np.nan
    _block(M, "openset_operating_point", openset_op)

    evset = set(evcats)
    cm = (np.array([n in evset for n in NM], bool) & (S >= 0.05)) if len(NM) else np.zeros(0, bool)

    def reliability():
        if not cm.any():
            return "no_detections_on_shared_classes"
        conf, corr = S[cm], TP[cm]
        M["ECE"] = ece_score(conf, corr)
        M["Brier"] = float(np.mean((conf - corr) ** 2))
        M["NLL"] = float(-np.mean(corr * np.log(np.clip(conf, 1e-6, 1)) + (~corr) * np.log(np.clip(1 - conf, 1e-6, 1))))
        for kind in RISK_TYPES:
            rk = np.concatenate([det_risk(it["p"], kind) for it in items])[cm]
            if not np.isfinite(rk).all():
                M[f"AURC_{kind}"] = M[f"EAURC_{kind}"] = np.nan; continue
            a, e, cv = risk_coverage(-rk, corr)
            M[f"AURC_{kind}"], M[f"EAURC_{kind}"] = a, e
            curves[f"rc_{kind}"] = cv
            if kind == "rcr":
                M["ECE_R"] = ece_score(1 - rk, corr)
        M["AURC"], M["AURC_R"] = M.get("AURC_msp"), M.get("AURC_rcr")
    _block(M, "reliability", reliability)

    def operational():
        fa = miss = ngt = 0
        per_bag = []
        for it in items:
            p = it["p"]
            ab = np.concatenate([p["boxes"][p["scores"] >= tk], p["ub"][p["us"] >= tu]])
            asc = np.concatenate([p["scores"][p["scores"] >= tk], p["us"][p["us"] >= tu]])
            if len(ab):
                ab = ab[nms(torch.from_numpy(ab), torch.from_numpy(asc), 0.5).numpy()]
            G = np.concatenate([it["kb"], it["ub"]]) if (len(it["kb"]) or len(it["ub"])) else np.zeros((0, 4), np.float32)
            iou = np_iou(ab, G)
            f = int((iou.max(1) < 0.5).sum()) if len(G) else len(ab)
            mi = int((iou.max(0) < 0.5).sum()) if len(ab) else len(G)
            fa += f; miss += mi; ngt += len(G)
            per_bag.append(dict(fa=f, miss=mi, ngt=len(G), neg=it["neg"], **{k: bag_risk(p, k) for k in RISK_TYPES}))
        M["FA_per_img"] = fa / len(items); M["miss_rate"] = miss / ngt if ngt else np.nan
        negs = [b for b in per_bag if b["neg"]]
        M["FA_per_clean_bag"] = float(np.mean([b["fa"] for b in negs])) if negs else np.nan
        bag_thr = thr.get("bag") or {}
        for kind in RISK_TYPES:
            for q in C.REFERRAL_BUDGETS:
                t = bag_thr.get(kind, {}).get(str(q))
                key = f"{kind}@{int(q * 100)}"
                if t is None or not np.isfinite(t):
                    M[f"ref_rate_{key}"] = M[f"miss_after_ref_{key}"] = M[f"fa_after_ref_{key}"] = np.nan; continue
                ref = np.array([np.isfinite(b[kind]) and b[kind] > t for b in per_bag])
                M[f"ref_rate_{key}"] = float(ref.mean())
                M[f"miss_after_ref_{key}"] = float(sum(b["miss"] for b, r in zip(per_bag, ref) if not r) / ngt) if ngt else np.nan
                M[f"fa_after_ref_{key}"] = float(np.mean([b["fa"] for b, r in zip(per_bag, ref) if not r])) if (~ref).any() else np.nan
        M["referral_rate"] = M.get("ref_rate_rcr@10", np.nan)
    _block(M, "operational", operational)
    if fps:
        M.update(fps)
    return M, curves


def compute_thresholds(preds, recs, known):
    """All operating points come from the SOURCE validation split (no target labels are ever used)."""
    items = build_eval_items(preds, recs, known)
    thr = dict(tau_known=0.5, tau_U=0.5, tau_R=0.5, bag={})
    if not items:
        return thr
    S = np.concatenate([it["p"]["scores"] for it in items]); TP = np.concatenate([it["tp"] for it in items])
    ngt = sum(len(it["kb"]) for it in items)
    if len(S) and ngt:
        o = np.argsort(-S); ctp = np.cumsum(TP[o]); prec = ctp / np.arange(1, len(o) + 1); rec = ctp / ngt
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9); thr["tau_known"] = float(np.clip(S[o][f1.argmax()], 0.05, 0.95))
    U = np.concatenate([it["p"]["U"] for it in items]); R = np.concatenate([it["p"]["R"] for it in items])
    m = (S >= thr["tau_known"]) & TP
    thr["tau_U"] = max(float(np.quantile(U[m], 0.95)) if m.sum() >= 5 else 0.5, 1e-3)   # 5% of correct known dets flagged
    sel = S >= thr["tau_known"]
    thr["tau_R"] = float(np.quantile(R[sel], 0.90)) if sel.sum() >= 5 else 0.5
    for kind in RISK_TYPES:
        br = np.array([bag_risk(it["p"], kind) for it in items], dtype=float)
        br = br[np.isfinite(br)]
        thr["bag"][kind] = {str(q): (float(np.quantile(br, 1 - q)) if len(br) >= 10 else None) for q in C.REFERRAL_BUDGETS}
    return thr


def paired_bootstrap(pA, pB, recs, known, metric="AP50", n=None, seed=0):
    """95% CI of Delta = metric(B) - metric(A); image-level resampling of the SAME images for both models."""
    n = n or C.BOOTSTRAP
    iA, iB = build_eval_items(pA, recs, known), build_eval_items(pB, recs, known)
    ids = sorted({it["i"] for it in iA} & {it["i"] for it in iB})
    if not ids:
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, p_le_0=np.nan)
    pos = {i: k for k, i in enumerate(ids)}

    def pack(items):
        D = defaultdict(lambda: [[], [], []]); G = defaultdict(lambda: np.zeros(len(ids)))
        for it in items:
            if it["i"] not in pos:
                continue
            k = pos[it["i"]]
            if metric == "U_AP50":
                p = it["p"]
                if len(p["ub"]):
                    t, _ = greedy_match(p["ub"], p["us"], it["ub"], 0.5)
                    D["u"][0] += list(p["us"]); D["u"][1] += list(t); D["u"][2] += [k] * len(t)
                G["u"][k] += len(it["ub"])
            else:
                for s, t, c in zip(it["p"]["scores"], it["tp"], it["names"]):
                    D[c][0].append(s); D[c][1].append(t); D[c][2].append(k)
                for c in it["kc"]:
                    G[c][k] += 1
        return {c: (np.array(v[0]), np.array(v[1], bool), np.array(v[2], int)) for c, v in D.items()}, G

    DA, GA = pack(iA); DB, GB = pack(iB)
    cats = [c for c in GA if GA[c].sum() > 0]
    if not cats:
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, p_le_0=np.nan)

    def score(D, G, w):
        vals = []
        for c in cats:
            ng = float((G[c] * w).sum())
            if ng <= 0:
                continue
            if c in D:
                s, t, im = D[c]; vals.append(ap_from(s, t, ng, w[im]))
            else:
                vals.append(0.0)
        return np.mean(vals) if vals else np.nan

    one = np.ones(len(ids))
    d0 = score(DB, GB, one) - score(DA, GA, one)
    rng = np.random.RandomState(seed); ds = []
    for _ in range(n):
        w = np.bincount(rng.randint(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        ds.append(score(DB, GB, w) - score(DA, GA, w))
    ds = np.array(ds)
    return dict(delta=float(d0), lo=float(np.nanpercentile(ds, 2.5)), hi=float(np.nanpercentile(ds, 97.5)),
                p_le_0=float(np.nanmean(ds <= 0)))
