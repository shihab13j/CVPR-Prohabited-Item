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


def pilot_check():
    """Seed-0 pilot: XGuard must beat the plain detector BEFORE the remaining seeds / ablations spend GPU time.
    Compares mean harmonized AP50 over targets with known classes and mean U-AP50 over targets with unknowns."""
    b_, x_ = run_path("six_base_s0"), run_path("six_xguard_s0")
    if not ((b_ / "DONE").exists() and (x_ / "DONE").exists()):
        return None
    def summary(rd):
        ap, uap = [], []
        for mf in sorted(rd.glob("metrics_*.json")):
            if mf.stem == "metrics_occsweep":
                continue
            a = (jload(mf, {}) or {}).get("all", {})
            if isinstance(a.get("AP50"), (int, float)) and not math.isnan(a["AP50"]):
                ap.append(a["AP50"])
            if isinstance(a.get("U_AP50"), (int, float)) and not math.isnan(a["U_AP50"]):
                uap.append(a["U_AP50"])
        return float(np.mean(ap)) if ap else float("nan"), float(np.mean(uap)) if uap else float("nan")
    (ba, bu), (xa, xu) = summary(b_), summary(x_)
    res = dict(base_AP50=ba, xguard_AP50=xa, base_U_AP50=bu, xguard_U_AP50=xu,
               known_ok=bool(xa >= ba - 0.01), unknown_ok=bool(xu > bu))
    res["passed"] = res["known_ok"] and res["unknown_ok"]
    jdump(res, WORK / "logs" / "pilot_check.json")
    print(f"\n🧪 PILOT (seed 0): mean AP50 base {100 * ba:.1f} vs XGuard {100 * xa:.1f} | "
          f"mean U-AP50 base {100 * bu:.1f} vs XGuard {100 * xu:.1f} -> {'PASS' if res['passed'] else 'NOT PASSED'}")
    return res


PILOT = None
_gate_on = MODE == "full" and C.PILOT_GATE and not C.PILOT_FORCE and \
    any(s["group"] == "core" and s["seed"] == 0 for s in SELECTED)
for spec in SELECTED:
    if _gate_on and not (spec["group"] == "core" and spec["seed"] == 0):
        PILOT = PILOT or pilot_check() or dict(passed=False, reason="seed-0 base / XGuard run did not finish")
        if not PILOT["passed"]:
            print("⛔ pilot not passed -> the remaining runs are NOT started, so no GPU time is wasted.\n"
                  "   Send me the PILOT line above (logs/pilot_check.json). Set C.PILOT_FORCE = True only to continue anyway.")
            break
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
    for rd in all_runs():
        sp, dep = jload(rd / "config.json", {}).get("spec"), jload(rd / "deploy.json", {})
        if sp and dep.get("sec_per_iter"):
            two_view = bool(sp.get("mcfg", {}).get("sira_cf") or sp.get("mcfg", {}).get("occ"))
            speed.setdefault((sp["kind"], two_view), []).append(dep["sec_per_iter"])
        if sp and dep.get("val_sec_per_epoch"):
            speed.setdefault("val", []).append(dep["val_sec_per_epoch"] * C.MAX_VAL / max(1, C.TEST_VAL or C.MAX_VAL))
        if sp and dep.get("fps"):
            speed.setdefault(("fps", sp["kind"]), []).append(dep["fps"])
    if not speed:
        return None
    sp_ = {k: float(np.median(v)) for k, v in speed.items()}
    old_seeds = (C.SEEDS_MAIN, C.SEEDS_ABL)
    C.SEEDS_MAIN, C.SEEDS_ABL = [0, 1, 2], [0]       # the FULL plan: 3 seeds for the core runs
    try:
        full_reg = [s for s in registry() if s["tier"] in TIERS]
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
        val_h = s["epochs"] * sp_.get("val", 60.0) / 3600 / max(1, C.VAL_EVERY)
        train_h = 0 if s["kind"] in ("owlvit", "gdino") else s["epochs"] * ipe * spi / 3600 * 1.05 + val_h
        rows.append(dict(tier=s["tier"], group=s["group"], run=s["name"], train_h=train_h, eval_h=n_eval / fps / 3600 * 1.3))
    df = pd.DataFrame(rows); df["total_h"] = df.train_h + df.eval_h
    cu = next((v for k, v in C.CU_PER_HOUR.items() if k in GPU_NAME), np.nan)
    summ = df.groupby(["tier", "group"]).agg(runs=("run", "count"), hours=("total_h", "sum")).reset_index()
    summ["compute_units"] = summ.hours * cu
    tot = df.groupby("tier").total_h.sum()
    print(f"\nProjected FULL plan for TIERS={TIERS} on {GPU_NAME} (measured speeds; Colab ≈ {cu} CU/h — verify on your Resources page):")
    display(summ)
    for t, h in tot.items():
        print(f"   tier {t}: {h:.1f} GPU-hours ≈ {h * cu:.0f} compute units")
    df.to_csv(WORK / "logs" / "full_plan_estimate.csv", index=False)
    return summ


if MODE == "test":
    PLAN = estimate_full_plan()
if MODE == "full" and PLATFORM == "colab":
    # every result is already on Drive; tables / figures / gates are built later with MODE='report' on a CPU runtime
    release_runtime("all selected Colab runs are finished or paused")
