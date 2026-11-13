# ==============================================================================
# 12. CORRECTNESS TESTS (run before any experiment; metric failures STOP the notebook)
#   * metric unit tests on hand-made toy cases (AP, one-to-one unknown recall, AUROC, E-AURC, zero-GT)
#   * CUS / pair-sampling integrity, AMP-dtype quantile, ontology integrity, CLIP text integrity
#   * reproducibility (test mode, GPU): same seed twice + interrupted/resumed run vs uninterrupted run
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
TESTS = OrderedDict()


def _toy_rec(boxes, cls, neg=False, subsets=()):
    return dict(boxes=np.array(boxes, np.float32).reshape(-1, 4), cls=list(cls), neg=neg, subsets=list(subsets), excluded=False)


def _toy_pred(boxes, scores, labels, ub=(), us=(), U=None):
    b = np.array(boxes, np.float32).reshape(-1, 4); s = np.array(scores, np.float32)
    ubb = np.array(ub, np.float32).reshape(-1, 4); uss = np.array(us, np.float32)
    U = np.zeros(len(s), np.float32) if U is None else np.array(U, np.float32)
    return dict(boxes=b, scores=s, labels=np.array(labels, np.int64), U=U, R=1 - s, E=-s, H=1 - s,
                ub=ubb, us=uss, uR=np.zeros(len(uss), np.float32), uE=np.zeros(len(uss), np.float32), uH=np.zeros(len(uss), np.float32))


def run_metric_tests():
    T = {}
    # 1. one-to-one matching: ONE detection covering TWO ground truths may match only one
    tp, gi = greedy_match(np.array([[0, 0, 10, 10]], np.float32), np.array([0.9]), np.array([[0, 0, 10, 10], [0, 0, 10, 9.5]], np.float32))
    T["greedy_one_to_one"] = bool(tp.sum() == 1)
    # 2. AP edge cases
    T["ap_perfect"] = abs(ap_from(np.array([0.9, 0.8]), np.array([True, True]), 2) - 1.0) < 1e-9
    T["ap_no_dets"] = ap_from(np.array([]), np.array([], bool), 3) == 0.0
    T["ap_zero_gt_is_nan"] = bool(np.isnan(ap_from(np.array([0.5]), np.array([False]), 0)))
    # 3. full evaluate() on a toy open-world benchmark (known = knife; gun is unknown)
    known = ["knife"]
    recs = [_toy_rec([[10, 10, 50, 50]], ["knife"]),
            _toy_rec([[100, 100, 150, 150], [100, 100, 150, 148]], ["gun", "gun"]),   # two unknown GTs, near-identical boxes
            _toy_rec([], [], neg=True, subsets=["negative"])]
    preds = [_toy_pred([[10, 10, 50, 50]], [0.9], [0], U=[0.05]),
             _toy_pred([], [], [], ub=[[100, 100, 150, 150]], us=[0.8]),                # ONE unknown detection
             _toy_pred([], [], [])]
    thr = dict(tau_known=0.5, tau_U=0.5, tau_R=0.5, bag={})
    M, _ = evaluate(preds, recs, known, thr)
    T["toy_AP50_is_1"] = abs(M["AP50"] - 1.0) < 1e-6
    T["toy_unknown_recall_one_to_one"] = abs(M["U_R50"] - 0.5) < 1e-9       # 1 det can hit only 1 of 2 GTs
    T["toy_status_ok"] = M["metric_status"]["closed_harmonized"] == "ok" and M["metric_status"]["unknown"] == "ok"
    T["toy_locAUROC_perfect"] = abs(M["locAUROC"] - 1.0) < 1e-9 and abs(M["locAUROC_flipped"]) < 1e-9
    # 4. zero-GT behaviour: only a clean bag -> not_applicable, never a crash / silent zero
    M0, _ = evaluate([preds[2]], [recs[2]], known, thr)
    T["zero_gt_not_applicable"] = M0["metric_status"]["closed_harmonized"].startswith("not_applicable") and \
        M0["metric_status"]["unknown"].startswith("not_applicable")
    # 5. selective risk: perfect ranking -> E-AURC = 0; random -> > 0
    corr = np.array([True] * 8 + [False] * 2)
    a, e, _ = risk_coverage(np.linspace(1, 0, 10), corr)
    T["eaurc_perfect_zero"] = abs(e) < 1e-6
    a2, e2, _ = risk_coverage(np.linspace(0, 1, 10), corr)
    T["eaurc_worst_positive"] = e2 > 0.1
    # 6. different-class pair sampling + CUS empty handling + fp32 quantile on AMP tensors
    y = torch.tensor([0, 0, 1, 1, 2])
    i, j = different_class_pairs(y, 200)
    T["pairs_always_different_class"] = bool((y[i] != y[j]).all())
    T["pairs_single_class_returns_None"] = different_class_pairs(torch.tensor([3, 3, 3]), 10) is None
    cus = CUS(16)
    T["cus_single_class_returns_None"] = cus.synth(torch.randn(4, 16), torch.tensor([1, 1, 1, 1]), 8) is None
    zs = cus.synth(torch.randn(6, 16), torch.tensor([0, 0, 1, 1, 2, 2]), 8)
    T["cus_shape"] = zs is not None and tuple(zs.shape) == (8, 16)
    try:
        torch.quantile(torch.randn(10).to(torch.bfloat16).float(), 0.25); T["quantile_fp32_cast"] = True
    except Exception:
        T["quantile_fp32_cast"] = False
    # 7. ontology integrity
    T["aliases_point_to_canonical"] = all(v in CLASS_ATTR for v in SEM_ALIAS.values()) and not (set(SEM_ALIAS) & set(SEM_ALIAS.values()))
    T["exact_track_ignores_alias"] = resolve_known("folding_knife", ["knife"], "exact") is None and \
        resolve_known("folding_knife", ["knife"], "harmonized") == "knife"
    data_classes = {c for r in ALL_REC for c in r["cls"]}
    T["all_data_classes_have_superclass"] = all(superclass(c) != "other" for c in data_classes)
    return T


TESTS.update(run_metric_tests())
failed = [k for k, v in TESTS.items() if not v]
print("metric / component tests:", "ALL PASSED" if not failed else f"FAILED: {failed}")
if failed:
    jdump(TESTS, WORK / "logs" / "unit_tests.json")
    raise AssertionError(f"correctness tests failed: {failed} -> fix before running experiments")

# CLIP text integrity (needed by STPB 't'): embeddings present, version + hash recorded
_tp = text_prototypes(["knife", "gun"])
TESTS["clip_text_available"] = _tp is not None
if _tp is not None:
    TESTS["clip_text_shapes"] = _tp["cls"].dim() == 2 and _tp["threat"].dim() == 1 and _tp["cls"].shape[1] == _tp["threat"].shape[0]
    assert TESTS["clip_text_shapes"], f"CLIP text prototypes have wrong shapes {tuple(_tp['cls'].shape)} / {tuple(_tp['threat'].shape)}"
# the text path of STPB must work even where no CLIP is available (synthetic 512-d text features)
_st = STPB(["knife", "gun"], 32, "vta", dict(cls=F.normalize(torch.randn(2, 512), dim=-1), threat=F.normalize(torch.randn(512), dim=0)))
_lg = _st.logits(torch.randn(5, 32)); _tp_ = _st.threat_proto()
TESTS["stpb_text_path"] = tuple(_lg.shape) == (5, 2) and tuple(_tp_.shape) == (32,)
assert TESTS["stpb_text_path"], "STPB text path broken"
TESTS["clip_text_info"] = dict(TEXT_INFO)
print("CLIP text prototypes:", "available" if _tp is not None else f"UNAVAILABLE ({TEXT_INFO['error']})")
if C.REQUIRE_TEXT and _tp is None:
    raise RuntimeError("REQUIRE_TEXT=True but CLIP text prototypes are unavailable (Internet off?)")

_FORCE_TIMEUP_AT = None


def reproducibility_test():
    """same seed twice -> parameter difference = hardware nondeterminism floor; interrupted+resumed vs continuous."""
    global _FORCE_TIMEUP_AT
    src = [r for r in REC.get("sixray", next(iter(REC.values()))) if r["split"] == "train" and not r["neg"]][:24]
    val = [r for r in REC.get("sixray", next(iter(REC.values()))) if r["split"] == "val"][:8]
    known = order_classes({c for r in src for c in r["cls"]})
    spec = dict(name="_repro", kind="xguard", mcfg=variant("xguard"), seed=123, init=C.INIT)
    old = (C.TEST_ITERS, C.BATCH)
    C.TEST_ITERS, C.BATCH = 6, 2
    outs = []
    try:
        for tag, stop in [("a", None), ("b", None), ("c", 3)]:
            rd = WORK / "logs" / f"_repro_{tag}"
            shutil.rmtree(rd, ignore_errors=True)
            _FORCE_TIMEUP_AT = stop
            try:
                m, _ = train_model(spec, rd, src, known, val, 1)
            except TimeUp:
                _FORCE_TIMEUP_AT = None
                m, _ = train_model(spec, rd, src, known, val, 1)     # resume from the saved full checkpoint
            outs.append(torch.cat([p.detach().float().flatten().cpu() for p in m.parameters()]))
            shutil.rmtree(rd, ignore_errors=True)
    finally:
        _FORCE_TIMEUP_AT = None
        C.TEST_ITERS, C.BATCH = old
    d_seed = float((outs[0] - outs[1]).abs().max()); d_resume = float((outs[0] - outs[2]).abs().max())
    return dict(same_seed_max_param_diff=d_seed, resume_vs_continuous_max_param_diff=d_resume,
                passed=bool(d_resume <= max(10 * d_seed, 1e-3)))


if MODE == "test" or SMOKE:
    TESTS["reproducibility"] = reproducibility_test()
    print("reproducibility:", TESTS["reproducibility"])
jdump(TESTS, WORK / "logs" / "unit_tests.json")
