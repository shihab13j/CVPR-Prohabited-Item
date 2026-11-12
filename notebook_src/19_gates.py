# ==============================================================================
# 18. SCIENTIFIC RELEASE GATES (automatic). A failed gate is logged with its reason; results that fail a
#     gate must not go into the manuscript until the cause is fixed.
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
jdump(TESTS, WORK / "logs" / f"unit_tests_{MODE}.json")
GATES = OrderedDict()


def gate(name, ok, detail, warn=False):
    GATES[name] = dict(status="PASS" if ok else ("WARN" if warn else "FAIL"), detail=detail)


# 1 dataset integrity
mism = VERIFICATION[(VERIFICATION.status == "MISMATCH")] if len(VERIFICATION) else VERIFICATION
mism_official = mism[mism.dataset != "sixray"] if len(mism) else mism
gate("dataset_integrity", len(mism_official) == 0,
     "train/val/test disjoint by ORIGINAL image id (asserted); official-size check: "
     + ("; ".join(f"{r.dataset}.{r.quantity} {r.observed}≠{r.official}" for r in mism.itertuples()) or "all match")
     + ". SIXray is a mirror split (documented), not the official one.")
# 2 ontology integrity
gate("ontology_integrity", TESTS.get("aliases_point_to_canonical") and TESTS.get("exact_track_ignores_alias") and TESTS.get("all_data_classes_have_superclass"),
     f"ontology {ONTOLOGY_VERSION}; aliases {sorted(SEM_ALIAS)}; exact / harmonized / hierarchical tracks separated")
# 3 CLIP integrity
text_runs = []
for rd in RUNS_DIR.glob("*"):
    sp = jload(rd / "config.json", {}).get("spec", {})
    st_ = STPB_ALIASES.get(sp.get("mcfg", {}).get("stpb", "none"), sp.get("mcfg", {}).get("stpb", "none"))
    if "t" in str(st_) and st_ != "none" and (rd / "train_meta.json").exists():
        text_runs.append((rd.name, jload(rd / "train_meta.json", {}).get("text_available")))
bad_text = [n for n, a in text_runs if a is False]
gate("clip_integrity", TESTS.get("clip_text_available") and not bad_text,
     f"CLIP {TEXT_INFO.get('source')} transformers={TEXT_INFO.get('transformers')} sha={TEXT_INFO.get('embedding_sha')}; "
     f"runs requesting text but trained WITHOUT it: {bad_text or 'none'}")
# 4 CUS integrity
skips = {}
for rd in RUNS_DIR.glob("*"):
    m_ = jload(rd / "train_meta.json", {})
    if m_.get("steps"):
        skips[rd.name] = m_.get("unk_skipped", 0) / max(1, m_["steps"])
heavy = {k: round(v, 3) for k, v in skips.items() if v > 0.2}
gate("cus_integrity", all(TESTS.get(k) for k in ["pairs_always_different_class", "pairs_single_class_returns_None",
                                                 "cus_single_class_returns_None", "cus_shape", "quantile_fp32_cast"]) and not heavy,
     f"different-class pairs guaranteed, fp32 quantiles, empty batches handled; runs skipping >20% of unknown steps: {heavy or 'none'}")
# 5 metric correctness
mt = [k for k in ["greedy_one_to_one", "ap_perfect", "ap_no_dets", "ap_zero_gt_is_nan", "toy_AP50_is_1", "toy_unknown_recall_one_to_one",
                  "toy_locAUROC_perfect", "zero_gt_not_applicable", "eaurc_perfect_zero", "eaurc_worst_positive"]]
evfail = []
for rd in RUNS_DIR.glob("*"):
    for mf in rd.glob("metrics_*.json"):
        for sub, M in (jload(mf, {}) or {}).items():
            if isinstance(M, dict):
                evfail += [f"{rd.name}/{mf.stem}/{sub}:{k}" for k, v in M.get("metric_status", {}).items() if str(v).startswith("evaluator_failure")]
gate("metric_correctness", all(TESTS.get(k) for k in mt) and not evfail, f"toy-case unit tests passed; evaluator failures: {evfail[:5] or 'none'}")
# 6 training reproducibility
rep = TESTS.get("reproducibility") or (jload(WORK / "logs" / "unit_tests_test.json", {}) or {}).get("reproducibility")
gate("training_reproducibility", bool(rep and rep.get("passed")),
     f"{rep}" if rep else "run MODE='test' first (it runs the same-seed and interrupted-resume checks)", warn=not rep)
# 7 fairness documentation
undoc = [rd.name for rd in RUNS_DIR.glob("*") if (rd / "DONE").exists() and not (rd / "config.json").exists()]
gate("fairness_documented", not undoc,
     "every run stores init/pretraining, epochs, resolution, prompts (oracle vocabulary flagged), thresholds from SOURCE val only"
     + (f"; undocumented: {undoc}" if undoc else ""))
# 8 evidence completeness
need = [s for s in REGISTRY if s["tier"] == 1]
done = [s for s in need if (RUNS_DIR / s["name"] / "DONE").exists()]
seeds_core = len({s["seed"] for s in need if s["tag"] == "six_xguard" and (RUNS_DIR / s["name"] / "DONE").exists()})
gate("evidence_complete", MODE == "full" and len(done) == len(need) and seeds_core >= 3,
     f"tier-1 runs finished {len(done)}/{len(need)}; core seeds {seeds_core}/3" + ("" if MODE == "full" else " (test mode: not applicable)"),
     warn=MODE != "full")
# 9 artifact integrity: every finished run's fingerprint matches the CURRENT code/data
stale = []
for s in SELECTED:
    rd = RUNS_DIR / s["name"]
    if (rd / "DONE").exists() and s["kind"] not in ("owlvit", "gdino", "yolo"):
        tr, kn = train_set(s)
        fp = fingerprints(s, tr, val_set(s), kn); old = jload(rd / "fingerprint.json", {})
        if old.get("train_fp") != fp["train_fp"] or old.get("eval_fp") != fp["eval_fp"]:
            stale.append(s["name"])
errs = RUN_STATUS[RUN_STATUS.status.str.startswith("error")].run.tolist() if len(RUN_STATUS) else []
gate("artifact_integrity", not stale and not errs, f"stale runs: {stale or 'none'}; errored runs: {errs or 'none'}")

GATE_DF = pd.DataFrame(GATES).T
jdump(GATES, WORK / f"gates_{MODE}.json")
display(GATE_DF)
n_fail = (GATE_DF.status == "FAIL").sum()
print(f"\n{'✅ all gates passed' if n_fail == 0 else f'❌ {n_fail} gate(s) FAILED'} (MODE={MODE})")
if MODE == "test":
    print("If every gate is PASS (evidence = WARN in test mode is expected), set MODE='full', choose TIERS, Run all.")
