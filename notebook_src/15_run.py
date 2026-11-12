# ==============================================================================
# 14. RUN (resumable; finished runs with a matching fingerprint are skipped)
#   test mode: every selected configuration runs C.TEST_ITERS iterations on REAL data, then the measured
#   speeds are projected to the full experiment plan (hours + Colab compute units per tier).
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
STATUS = OrderedDict()
need_six = [s for s in SELECTED if "sixray" in s["train"]]
if "sixray" not in AVAILABLE and need_six:
    raise RuntimeError("SIXray has no box annotations in the pack -> rebuild the pack with khanhbtq99/sixray attached.")
for spec in SELECTED:
    if time_left_h() < 0.5:
        print(f"⏱  {elapsed_h():.1f} h used -> stopping. Re-run the notebook to continue (finished runs are kept on Drive).")
        break
    try:
        STATUS[spec["name"]] = run_experiment(spec)
    except TimeUp as e:
        STATUS[spec["name"]] = "paused"; print("⏱ ", e); break
    except Exception as e:
        traceback.print_exc()
        STATUS[spec["name"]] = f"error: {type(e).__name__}: {str(e)[:200]}"
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
RUN_STATUS = pd.DataFrame([dict(run=k, status=v) for k, v in STATUS.items()])
display(RUN_STATUS)
RUN_STATUS.to_csv(WORK / "logs" / f"run_status_{MODE}.csv", index=False)
errors = RUN_STATUS[RUN_STATUS.status.str.startswith("error")] if len(RUN_STATUS) else RUN_STATUS


def estimate_full_plan():
    """Project measured test-mode speeds to the full plan (all tiers) on this GPU."""
    rows, speed = [], {}
    for rd in RUNS_DIR.glob("*"):
        sp, dep = jload(rd / "config.json", {}).get("spec"), jload(rd / "deploy.json", {})
        if sp and dep.get("sec_per_iter"):
            two_view = bool(sp.get("mcfg", {}).get("sira_cf") or sp.get("mcfg", {}).get("occ"))
            speed.setdefault((sp["kind"], two_view), []).append(dep["sec_per_iter"])
        if sp and dep.get("fps"):
            speed.setdefault(("fps", sp["kind"]), []).append(dep["fps"])
    if not speed:
        return None
    sp_ = {k: float(np.median(v)) for k, v in speed.items()}
    old_seeds = (C.SEEDS_MAIN, C.SEEDS_ABL)
    C.SEEDS_MAIN, C.SEEDS_ABL = [0, 1, 2], [0]       # the FULL plan: 3 seeds for the core runs
    try:
        full_reg = registry()
    finally:
        C.SEEDS_MAIN, C.SEEDS_ABL = old_seeds
    for s in full_reg:
        tr, _ = train_set(s)
        ipe = math.ceil(len(tr) / C.BATCH)
        two = bool(s.get("mcfg", {}).get("sira_cf") or s.get("mcfg", {}).get("occ"))
        spi = sp_.get((s["kind"], two)) or sp_.get(("xguard", two)) or np.nan
        n_eval = sum(len([r for r in split_recs(t, "test")]) if s["full_eval"] else min(C.MAX_EVAL_ABL, len(split_recs(t, "test")))
                     for t in s["targets"] if t in AVAILABLE)
        fps = sp_.get(("fps", s["kind"])) or sp_.get(("fps", "xguard")) or np.nan
        train_h = 0 if s["kind"] in ("owlvit", "gdino") else s["epochs"] * ipe * spi / 3600 * 1.05
        rows.append(dict(tier=s["tier"], group=s["group"], run=s["name"], train_h=train_h, eval_h=n_eval / fps / 3600 * 1.3))
    df = pd.DataFrame(rows); df["total_h"] = df.train_h + df.eval_h
    cu = next((v for k, v in C.CU_PER_HOUR.items() if k in GPU_NAME), np.nan)
    summ = df.groupby(["tier", "group"]).agg(runs=("run", "count"), hours=("total_h", "sum")).reset_index()
    summ["compute_units"] = summ.hours * cu
    tot = df.groupby("tier").total_h.sum()
    print(f"\nProjected FULL plan on {GPU_NAME} (measured speeds; Colab ≈ {cu} CU/h — verify on your Resources page):")
    display(summ)
    for t, h in tot.items():
        print(f"   tier {t}: {h:.1f} GPU-hours ≈ {h * cu:.0f} compute units")
    df.to_csv(WORK / "logs" / "full_plan_estimate.csv", index=False)
    return summ


if MODE == "test":
    PLAN = estimate_full_plan()
