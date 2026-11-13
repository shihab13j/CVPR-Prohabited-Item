# ==============================================================================
# 15. TABLES T1-T12 (+ LOCO, LODO, literature) -> tables/*.csv and *.tex
#   cells: mean±sd over finished seeds (n shown in T11); "n/a" = not applicable (metric_status says why);
#   "—" = run not finished / not selected. Nothing is ever filled in by hand.
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
def load_results():
    rows = []
    for rd in all_runs():
        if not (rd / "DONE").exists():
            continue
        cfg, dep = jload(rd / "config.json", {}), jload(rd / "deploy.json", {})
        spec = cfg.get("spec", {})
        for mf in rd.glob("metrics_*.json"):
            tgt = mf.stem.replace("metrics_", "")
            if tgt == "occsweep":
                for r_, v in jload(mf, {}).items():
                    rows.append(dict(run=rd.name, tag=spec.get("tag"), group=spec.get("group"), seed=spec.get("seed"),
                                     target="sixray", subset=f"occ{r_}", AP50=v))
                continue
            for sub, M in jload(mf, {}).items():
                row = dict(run=rd.name, tag=spec.get("tag"), group=spec.get("group"), seed=spec.get("seed"), target=tgt, subset=sub)
                row.update({k: v for k, v in M.items() if not isinstance(v, (dict, list))})
                row.update({f"dep_{k}": v for k, v in dep.items() if not isinstance(v, (dict, list))})
                rows.append(row)
    return pd.DataFrame(rows)


RES = load_results()
print(f"{RES.run.nunique() if len(RES) else 0} finished runs in {[str(d) for d in RUNS_DIRS]}")
RES.to_csv(WORK / "tables" / f"all_results_long_{MODE}.csv", index=False)
TAB = WORK / "tables" / MODE
TAB.mkdir(parents=True, exist_ok=True)


def val(tag, target, metric, subset="all", fmt=True, scale=100.0):
    if not len(RES) or "tag" not in RES:
        return "—" if fmt else np.nan
    d = RES[(RES.tag == tag) & (RES.target == target) & (RES.subset == subset)]
    if not len(d):
        return "—" if fmt else np.nan
    if metric not in d or d[metric].dropna().empty:
        return "n/a" if fmt else np.nan
    v = d[metric].dropna().astype(float) * scale
    if not fmt:
        return float(v.mean())
    return f"{v.mean():.1f}±{v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def num(x):
    try:
        return float(str(x).split("±")[0])
    except ValueError:
        return np.nan


def merge_external(df, table):
    if C.EXTERNAL_RESULTS and os.path.exists(C.EXTERNAL_RESULTS):
        ext = pd.read_csv(C.EXTERNAL_RESULTS)
        ext = ext[ext.table == table] if "table" in ext else ext.iloc[0:0]
        for m, g in ext.groupby("method"):
            row = {c: "—" for c in df.columns}; row.update({r.column: f"{r.value}" for r in g.itertuples() if r.column in df.columns})
            df.loc[f"{m} (reported)"] = row
    return df


def save_table(df, name, caption):
    df = merge_external(df, name)
    df.to_csv(TAB / f"{name}.csv")
    try:
        tex = df.to_latex(escape=True, caption=caption, label=f"tab:{name}")
    except Exception:
        tex = "% " + caption + "\n" + df.to_string()
    (TAB / f"{name}.tex").write_text(tex)
    display(Markdown(f"**{name}** — {caption}")); display(df)
    return df


NICE = {"six_base": "Faster R-CNN R50-FPN (base)", "six_fcos": "FCOS", "six_retinanet": "RetinaNet", "six_yolo11s": "YOLO11-s",
        "six_rtdetr-l": "RT-DETR-L", "six_owlvit": "OWL-ViT (zero-shot, name-free)", "six_gdino": "Grounding-DINO (zero-shot, name-free)",
        "six_owlvit-oracle": "OWL-ViT (ORACLE vocabulary)†", "six_gdino-oracle": "Grounding-DINO (ORACLE vocabulary)†",
        "six_xguard": "XGuard (composite)", "six_base-occaug": "Faster R-CNN + occlusion aug (control)"}
MAIN = ["six_base", "six_base-occaug", "six_fcos", "six_retinanet", "six_yolo11s", "six_rtdetr-l", "six_owlvit", "six_gdino", "six_owlvit-oracle",
        "six_gdino-oracle", "six_xguard"]
TG = [t for t in ["pidray", "opixray", "hixray", "clcxray"] if t in AVAILABLE]

# ---- T0 literature (qualitative; VERIFY every cell against the cited papers before submission) ----
save_table(pd.DataFrame([
    ["CHR (SIXray)", "–", "–", "partial", "–", "–"], ["DOAM (OPIXray)", "–", "–", "✓", "–", "–"],
    ["LIM (HiXray)", "–", "–", "partial", "–", "–"], ["SDANet (PIDray)", "–", "–", "✓ (hidden)", "–", "–"],
    ["OVXD", "partial", "✓ (named novel classes)", "–", "–", "–"], ["STING-BEE", "✓", "multimodal / zero-shot", "–", "–", "–"],
    ["XGuard (ours)", "✓", "✓ (name-free)", "✓", "✓", "✓"]],
    columns=["Method", "Domain shift", "Unknown class", "Occlusion", "Risk / referral", "Compound"]).set_index("Method"),
    "T0_literature", "Qualitative positioning (verify each cell against the papers).")

# ---- T1 dataset statistics, splits, ontology ----
rows = {}
for k, v in STATS.items():
    ver = VERIFICATION[VERIFICATION.dataset == k] if len(VERIFICATION) else VERIFICATION
    rows[k] = {"split origin": SPLIT_ORIGIN.get(k, "as shipped"),
               "classes": len(CLASS_AUDIT[CLASS_AUDIT.dataset == k]),
               **{f"{sp} img": v["splits"][sp]["images"] for sp in ["train", "val", "test"]},
               **{f"{sp} obj": v["splits"][sp]["instances"] for sp in ["train", "test"]},
               "test subsets": ", ".join(f"{a}:{b}" for a, b in v["splits"]["test"]["subsets"].items()),
               "excluded (leakage)": AUDIT_SUMMARY.get("excluded_by_dataset", {}).get(k, 0),
               "official-size check": ", ".join(f"{r.quantity}:{r.status}" for r in ver.itertuples()) or "—",
               "lineage": f"{LINEAGE_LOG.get(k, {}).get('files', '?')} files → {LINEAGE_LOG.get(k, {}).get('unique_ids', '?')} ids"}
save_table(pd.DataFrame(rows).T, "T1_datasets", "Dataset statistics, split provenance, leakage exclusions and verification against official sizes.")

# ---- T2 in-domain ----
rows = {NICE.get(t, t): {m: val(t, "sixray", m) for m in ["AP", "AP50", "AP75", "AR100", "APs", "APm", "APl"]} for t in MAIN}
for ds in ["opixray", "pidray", "hixray"]:
    for v in ["base", "xguard"]:
        rows[f"[{ds}] {v}"] = {m: val(f"in-{ds}_{v}", ds, m) for m in ["AP", "AP50", "AP75", "AR100", "APs", "APm", "APl"]}
save_table(pd.DataFrame(rows).T, "T2_in_domain", "P1 in-domain known-object detection on the SIXray (mirror) test split, COCO metrics (%).")

# ---- T3 cross-dataset generalization (three tracks) ----
rows = {}
for t in MAIN + ["six_sira"]:
    r = {}
    for g in TG:
        r[f"{g} xAP50"], r[f"{g} AP50"], r[f"{g} hAP50"] = val(t, g, "xAP50"), val(t, g, "AP50"), val(t, g, "hAP50")
    a = [val(t, g, "AP50", fmt=False) for g in TG]
    r["mean AP50 (harmonized)"] = f"{np.nanmean(a):.1f}" if np.isfinite(a).any() else "n/a"
    rows[NICE.get(t, t)] = r
save_table(pd.DataFrame(rows).T, "T3_cross_domain", "P2 single-source generalization (train SIXray only). xAP50 exact names, AP50 harmonized aliases, hAP50 superclass.")

# ---- T4 unknown detection and false alarms ----
rows = {}
for t in MAIN + ["six_sira+oer+stpb+cus"]:
    r = {}
    for g in [x for x in ["pidray", "hixray", "opixray"] if x in AVAILABLE]:
        for m, lab in [("U_AP50", "U-AP50"), ("U_R50", "U-R50"), ("U_R50_op", "U-R50@τ"), ("locAUROC", "locAUROC"),
                       ("locCoverage", "loc-cov"), ("FA_per_img", "FA/img")]:
            r[f"{g} {lab}"] = val(t, g, m, scale=1 if m == "FA_per_img" else 100)
        r[f"{g} AOSE"] = val(t, g, "AOSE", scale=1)
    r["SIXray FA/clean bag"] = val(t, "sixray", "FA_per_clean_bag", "negative", scale=1)
    rows[NICE.get(t, t)] = r
save_table(pd.DataFrame(rows).T, "T4_unknown", "P5 unknown threats (never named in training). locAUROC only over localized objects (loc-cov). †oracle = unseen names given.")

# LOCO
rows = {}
for c in LOCO_CLASSES:
    for v in ["base", "xguard"]:
        rows[f"{c} | {v}"] = {m: val(f"loco-{c}_{v}", "sixray", k) for m, k in
                             [("Known AP50", "AP50"), ("U-AP50", "U_AP50"), ("U-R50", "U_R50"), ("locAUROC", "locAUROC"), ("AOSE", "AOSE")]}
if rows:
    df = pd.DataFrame(rows).T
    for v in ["base", "xguard"]:
        sub = df[[i.endswith(v) for i in df.index]].apply(lambda col: col.map(num))
        df.loc[f"mean | {v}"] = [f"{x:.1f}" if np.isfinite(x) else "n/a" for x in sub.mean()]
    save_table(df, "T4b_loco", "P4 leave-one-class-out on SIXray (folds = classes with enough train AND test objects).")

# ---- T5 occlusion robustness ----
rows = {}
for t in ["six_base", "six_base-occaug", "six_oer", "six_sira+oer", "occ-cutout", "six_xguard"]:
    r = {f"OL{i}": val(t, "opixray", "hAP50", f"OL{i}") for i in (1, 2, 3)}
    for r_ in C.OCC_SWEEP:
        r[f"synthetic occ {int(r_ * 100)}% AP50"] = val(t, "sixray", "AP50", f"occ{r_}")
    a0, a5 = val(t, "sixray", "AP50", f"occ{C.OCC_SWEEP[0]}", fmt=False), val(t, "sixray", "AP50", f"occ{C.OCC_SWEEP[-1]}", fmt=False)
    r["degradation 0→max ↓"] = f"{(a0 - a5) / a0:.3f}" if np.isfinite(a0) and np.isfinite(a5) and a0 else "n/a"
    rows[NICE.get(t, t)] = r
save_table(pd.DataFrame(rows).T, "T5_occlusion", "P6 occlusion: official OPIXray OL1-3 (if lists provided) + deterministic synthetic occlusion sweep on SIXray test.")

# ---- T6 compound shift ----
rows = {}
for t in ["six_base", "six_base-occaug", "six_xguard", "six_gdino", "six_owlvit"]:
    rows[f"PIDray-hidden, unseen classes | {NICE.get(t, t)}"] = {
        "U-AP50": val(t, "pidray", "U_AP50", "hidden"), "U-R50": val(t, "pidray", "U_R50", "hidden"),
        "miss-rate ↓": val(t, "pidray", "miss_rate", "hidden"), "FA/img ↓": val(t, "pidray", "FA_per_img", "hidden", scale=1)}
for v in ["base", "xguard"]:
    rows[f"OPIXray, knife never seen (LOCO-knife) | {v}"] = {
        "U-AP50": val(f"loco-knife_{v}", "opixray", "U_AP50"), "U-R50": val(f"loco-knife_{v}", "opixray", "U_R50"),
        "miss-rate ↓": val(f"loco-knife_{v}", "opixray", "miss_rate"), "FA/img ↓": val(f"loco-knife_{v}", "opixray", "FA_per_img", scale=1)}
save_table(pd.DataFrame(rows).T, "T6_compound", "P7 compound shift: unseen scanner × unseen class × concealment (PIDray hidden).")

# ---- T7 component ablation (independent, cumulative, leave-one-out) ----
ABL = [("six_base", "", "", "", "", ""), ("six_sira", "✓", "", "", "", ""), ("six_oer", "", "✓", "", "", ""),
       ("six_stpb", "", "", "✓", "", ""), ("six_sira+oer", "✓", "✓", "", "", ""), ("six_sira+oer+stpb", "✓", "✓", "✓", "", ""),
       ("six_sira+oer+stpb+cus", "✓", "✓", "✓", "✓", ""), ("six_xguard", "✓", "✓", "✓", "✓", "✓"),
       ("six_xguard-sira", "", "✓", "✓", "✓", "✓"), ("six_xguard-oer", "✓", "", "✓", "✓", "✓"),
       ("six_xguard-cus", "✓", "✓", "✓", "", "✓"), ("six_sira+oer+stpb+cus", "✓", "✓", "✓", "✓", "– (=full−RCR)")]


def abl_metrics(t):
    base = lambda m, g, sub="all": val("six_base", g, m, sub, fmt=False)
    me = lambda m, g, sub="all": val(t, g, m, sub, fmt=False)
    d_dom = me("AP50", "pidray") - base("AP50", "pidray")
    d_unk = me("U_AP50", "pidray") - base("U_AP50", "pidray")
    d_rc = me("AURC_rcr", "pidray") - base("AURC_msp", "pidray")
    f = lambda x: f"{x:+.1f}" if np.isfinite(x) else "n/a"
    return {"SIXray AP50": val(t, "sixray", "AP50"), "PIDray AP50 (domain)": val(t, "pidray", "AP50"),
            "PIDray U-AP50": val(t, "pidray", "U_AP50"), "occ 50% AP50": val(t, "sixray", "AP50", "occ0.5"),
            "AURC rcr ↓": val(t, "pidray", "AURC_rcr"), "ΔAP_domain": f(d_dom), "ΔU-AP": f(d_unk), "ΔAURC ↓": f(d_rc)}


rows = {}
for t, *flags in ABL:
    key = f"{'+'.join(n for n, f in zip(['SIRA', 'OER', 'STPB', 'CUS', 'RCR'], flags) if f == '✓') or 'base'}" + (" (full−RCR)" if flags[-1].startswith("–") else "")
    rows[key] = {"SIRA": flags[0], "OER": flags[1], "STPB": flags[2], "CUS": flags[3], "RCR": flags[4], **abl_metrics(t)}
save_table(pd.DataFrame(rows).T, "T7_ablation", "Component ablation (single seed, fixed eval subsets). Δ relative to the matched base detector.")

# ---- T8 prototype sources ----
rows = {}
for t, lab in [("proto-v", "visual"), ("proto-t", "text"), ("proto-a", "attribute"), ("proto-vt", "visual+text"),
               ("proto-va", "visual+attribute"), ("six_xguard", "visual+text+attribute (full)")]:
    rows[lab] = {"SIXray AP50": val(t, "sixray", "AP50"), "PIDray AP50": val(t, "pidray", "AP50"), "PIDray U-AP50": val(t, "pidray", "U_AP50"),
                 "HiXray U-AP50": val(t, "hixray", "U_AP50"), "locAUROC (PIDray)": val(t, "pidray", "locAUROC")}
save_table(pd.DataFrame(rows).T, "T8_prototype", "STPB prototype-source ablation inside the full model (text = CLIP; see CLIP integrity in gates).")

# ---- T9 synthetic unknown generation ----
rows = {}
for t, lab in [("unk-none", "none (U = o·s_t·(1−s_k))"), ("unk-gaussian", "Gaussian feature noise"), ("unk-mixup", "feature mixup"),
               ("unk-slerp", "manifold (slerp) interpolation"), ("six_xguard", "CUS (factorized, ours)"), ("unk-cus-noent", "CUS without entropy loss")]:
    log_ = pd.read_csv(run_path(f"{t}_s0") / "train_log.csv") if (run_path(f"{t}_s0") / "train_log.csv").exists() else pd.DataFrame()
    rows[lab] = {"SIXray known AP50": val(t, "sixray", "AP50"), "PIDray U-AP50": val(t, "pidray", "U_AP50"), "PIDray U-R50": val(t, "pidray", "U_R50"),
                 "HiXray U-AP50": val(t, "hixray", "U_AP50"), "locAUROC": val(t, "pidray", "locAUROC"),
                 "synthetic max-known cos": f"{log_.synth_max_known_cos.dropna().iloc[-5:].mean():.3f}" if "synth_max_known_cos" in log_ and log_.synth_max_known_cos.notna().any() else "n/a",
                 "real max-known cos": f"{log_.real_max_known_cos.dropna().iloc[-5:].mean():.3f}" if "real_max_known_cos" in log_ and log_.real_max_known_cos.notna().any() else "n/a"}
save_table(pd.DataFrame(rows).T, "T9_unknown_generation", "Synthetic-unknown strategies under the same protocol (+ known-support similarity of generated features).")

# ---- T10 reliability / referral ----
rows = {}
for t in ["six_base", "six_xguard"]:
    for k in RISK_TYPES:
        rows[f"{NICE.get(t, t)} | {k}"] = {
            "AURC ↓": val(t, "pidray", f"AURC_{k}"), "E-AURC ↓": val(t, "pidray", f"EAURC_{k}"),
            **{f"referral @{int(q * 100)}%": val(t, "pidray", f"ref_rate_{k}@{int(q * 100)}") for q in C.REFERRAL_BUDGETS},
            **{f"missed threats after ref @{int(q * 100)}% ↓": val(t, "pidray", f"miss_after_ref_{k}@{int(q * 100)}") for q in C.REFERRAL_BUDGETS},
            **{f"FA/bag after ref @{int(q * 100)}% ↓": val(t, "pidray", f"fa_after_ref_{k}@{int(q * 100)}", scale=1) for q in C.REFERRAL_BUDGETS}}
    rows[f"{NICE.get(t, t)} | calibration"] = {"AURC ↓": f"ECE {val(t, 'pidray', 'ECE')} / ECE(1−R) {val(t, 'pidray', 'ECE_R')}",
                                               "E-AURC ↓": f"Brier {val(t, 'pidray', 'Brier')}"}
save_table(pd.DataFrame(rows).T.fillna(""), "T10_reliability", "Risk scores compared on PIDray (unseen scanner): detection-level AURC/E-AURC and bag-level referral at budgets fixed on SIXray val.")

# ---- T11 multi-seed statistics + paired bootstrap ----
rows = {}
for t in ["six_base", "six_base-occaug", "six_xguard"]:
    n = RES[(RES.tag == t)].seed.nunique() if len(RES) else 0
    rows[f"{NICE.get(t, t)} (n={n} seeds)"] = {f"{g} {m}": val(t, g, m) for g in ["sixray"] + TG for m in ["AP50", "U_AP50"]}
save_table(pd.DataFrame(rows).T, "T11_seeds", "Mean ± SD over independent seeds.")
rows = {}
for t in ["sixray"] + TG:
    pa, pb = run_path("six_base_s0") / f"preds_{t}.pkl", run_path("six_xguard_s0") / f"preds_{t}.pkl"
    if pa.exists() and pb.exists():
        known = jload(run_path("six_xguard_s0") / "config.json")["known"]
        rec_t = eval_set(t, True)
        for metric in ["AP50", "U_AP50"]:
            ci = paired_bootstrap(load_preds(pa, rec_t)[0], load_preds(pb, rec_t)[0], rec_t, known, metric)
            rows[f"{t} {metric}"] = {"Δ (XGuard−base)": f"{100 * ci['delta']:+.1f}" if np.isfinite(ci["delta"]) else "n/a",
                                     "95% CI": f"[{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}]" if np.isfinite(ci["lo"]) else "n/a",
                                     "P(Δ≤0)": f"{ci['p_le_0']:.3f}" if np.isfinite(ci["p_le_0"]) else "n/a"}
save_table(pd.DataFrame(rows).T, "T11b_bootstrap", f"Paired image-level bootstrap ({C.BOOTSTRAP} resamples), same test images for both models.")

# ---- T12 efficiency ----
rows = {}
for t in MAIN + ["six_sira", "six_oer", "six_sira+oer+stpb+cus"]:
    d = RES[(RES.tag == t)] if len(RES) else pd.DataFrame()
    g = lambda k, f="{:.1f}": (f.format(d[k].dropna().astype(float).mean()) if len(d) and k in d and d[k].dropna().size else "—")
    rows[NICE.get(t, t)] = {"Params (M)": g("dep_params_M"), "GFLOPs": g("dep_GFLOPs"), "FPS": g("dep_fps"), "ms/img": g("dep_ms_per_img"),
                            "train min": g("dep_train_minutes", "{:.0f}")}
save_table(pd.DataFrame(rows).T, "T12_efficiency", f"Efficiency on {GPU_NAME}, input {C.IMG}px, AMP {AMP_DTYPE}.")

# ---- LODO ----
rows = {}
for held in ["sixray", "opixray", "pidray", "hixray"]:
    for v in ["base", "xguard"]:
        rows[f"→ {held} | {v}"] = {m: val(f"lodo-{held}_{v}", held, k) for m, k in [("AP50", "AP50"), ("hAP50", "hAP50"), ("U-AP50", "U_AP50"), ("locAUROC", "locAUROC")]}
save_table(pd.DataFrame(rows).T, "T13_lodo", "P3 leave-one-domain-out (tier 3).")
