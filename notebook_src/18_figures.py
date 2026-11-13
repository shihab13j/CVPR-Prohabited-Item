# ==============================================================================
# 17. FIGURES 1-8 (PNG + PDF). Only Fig. 1 (architecture) is a drawing; every other figure is generated
#     from real model outputs of this run (no illustrative / fabricated detections).
#   1 architecture · 2 feature space by dataset · 3 qualitative known vs unknown · 4 occlusion curves
#   5 risk-coverage + referral · 6 CUS validity · 7 failure cases · 8 per-class cross-domain AP heatmap
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
wait_pack_all()          # qualitative figures read test images of every dataset
FIG = WORK / "figures" / MODE
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def savefig(name):
    fig_ = plt.gcf()           # paper figures carry their caption in LaTeX: drop the "Fig. N ..." titles from the image
    if getattr(fig_, "_suptitle", None) is not None and fig_._suptitle.get_text().startswith("Fig."):
        fig_._suptitle.set_text("")
    for ax_ in fig_.axes:
        if ax_.get_title().startswith("Fig."):
            ax_.set_title("")
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.show(); plt.close()


def box(ax, x, y, w, h, text, fc="#E8EEF7", ec="#34495E", fs=8, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, c="#34495E"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=c, lw=1.3))


def run_file(tag, name, seed=0):
    p = run_path(f"{tag}_s{seed}") / name
    return p if p.exists() else None


# ---------- Fig. 1 architecture ----------
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box(ax, 0.01, 0.42, 0.09, 0.16, "X-ray\nimage", fc="#EEEEEE")
box(ax, 0.13, 0.40, 0.12, 0.20, "ResNet-50\n+ FPN", fc="#DDEBF7")
box(ax, 0.28, 0.66, 0.16, 0.22, "SIRA\nF_L=LP(F), F_H=F−F_L\nG=σ(MLP[F_L,F_H])\n+ T_φ consistency", fc="#FCE4D6", fs=7)
box(ax, 0.28, 0.12, 0.16, 0.22, "RPN\n→ RoI tokens (7×7)", fc="#EEEEEE", fs=7)
box(ax, 0.48, 0.12, 0.16, 0.22, "OER\nA=qkᵀ/√d+λ·log r_i\nvisibility-supervised r_i", fc="#FCE4D6", fs=7)
box(ax, 0.48, 0.66, 0.16, 0.22, "STPB\np_c = Σ w_k p_c^k\nk ∈ {visual, text, attribute}", fc="#E2F0D9", fs=7)
box(ax, 0.68, 0.40, 0.13, 0.20, "RoI embedding z\nbox · class\nU=o·u(z)", fc="#DDEBF7", fs=7)
box(ax, 0.68, 0.72, 0.13, 0.18, "CUS (train only)\n[s_A, m_B, λg_A+(1−λ)g_B]", fc="#E2F0D9", fs=7)
box(ax, 0.84, 0.40, 0.15, 0.20, "RCR\nR=P(wrong|z)\nDetect / Refer", fc="#FFF2CC", fs=8, bold=True)
for a in [(0.10, 0.5, 0.13, 0.5), (0.25, 0.55, 0.28, 0.75), (0.25, 0.45, 0.28, 0.25), (0.44, 0.23, 0.48, 0.23),
          (0.64, 0.23, 0.68, 0.45), (0.64, 0.77, 0.68, 0.55), (0.745, 0.72, 0.745, 0.60), (0.81, 0.5, 0.84, 0.5), (0.36, 0.66, 0.36, 0.34)]:
    arrow(ax, *a)
ax.set_title("Fig. 1  XGuard architecture", y=-0.02)
savefig("fig1_architecture")

# ---------- Fig. 2 feature space coloured by dataset ----------
embs = {t: load_emb(t) for t in ["six_base", "six_sira", "six_xguard"]}
embs = {k: v for k, v in embs.items() if v is not None}
if embs:
    fig, axes = plt.subplots(1, len(embs), figsize=(4.6 * len(embs), 4.2), squeeze=False)
    for ax, (t, e) in zip(axes[0], embs.items()):
        Z = e["z"]; sel = np.random.RandomState(0).choice(len(Z), min(len(Z), 3000), replace=False)
        P2 = TSNE(2, init="pca", perplexity=min(30, max(5, len(sel) // 4)), random_state=0).fit_transform(
            Z[sel] / np.linalg.norm(Z[sel], axis=1, keepdims=True).clip(1e-6))
        for d in sorted(set(e["ds"])):
            m = e["ds"][sel] == d; ax.scatter(P2[m, 0], P2[m, 1], s=4, label=d, alpha=0.6)
        ax.set_title(NICE.get(t, t)); ax.legend(fontsize=7, markerscale=3); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Fig. 2  GT-object embeddings (t-SNE) coloured by dataset / scanner")
    savefig("fig2_feature_space")


# ---------- helpers for prediction-based figures ----------
def preds_of(tag, t, full=True):
    pf = run_file(tag, f"preds_{t}.pkl")
    return load_preds(pf, eval_set(t, full))[0] if pf else None


def draw(ax, r, p, kn, thr, title=""):
    im = np.asarray(load_rgb(r["path"], 1024).resize((r["w"], r["h"])))
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    if p is None:
        return
    for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):
        if s >= thr["tau_known"]:
            ax.add_patch(mpatches.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.3))
            ax.text(b[0], b[1], f"{kn[l]} {s:.2f}", color="lime", fontsize=6, va="bottom")
    for b, s in zip(p["ub"], p["us"]):
        if s >= thr["tau_U"]:
            ax.add_patch(mpatches.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="magenta", lw=1.3, ls="--"))
            ax.text(b[0], b[3], f"UNKNOWN {s:.2f}", color="magenta", fontsize=6, va="top")


def draw_gt(ax, r):
    for b, c in zip(r["boxes"], r["cls"]):
        ax.add_patch(mpatches.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec="yellow", lw=1.5))
        ax.text(b[0], b[1], c, color="yellow", fontsize=6, va="bottom")


def pick(tag_x, t, kind, known_x, thr, sub=None, k=1):
    px = preds_of(tag_x, t)
    if px is None:
        return []
    scored = []
    for i, r in enumerate(eval_set(t, True)):
        if px[i] is None or r.get("excluded") or (sub and sub not in r["subsets"]):
            continue
        p = px[i]
        A = np.concatenate([p["boxes"][p["scores"] >= thr["tau_known"]], p["ub"][p["us"] >= thr["tau_U"]]])
        G = r["boxes"]
        hit = np_iou(G, A).max(1) >= 0.5 if len(G) and len(A) else np.zeros(len(G), bool)
        unk = np.array([resolve_known(c, known_x) is None for c in r["cls"]], bool)
        if kind == "unknown_found":
            s = (hit & unk).sum() if unk.any() else -1
        elif kind == "known_found":
            s = (hit & ~unk).sum() - 0.1 * len(A) if (~unk).any() else -1
        elif kind == "missed":
            s = (~hit).sum() if len(G) else -1
        elif kind == "false_alarm":
            s = int((np_iou(A, G).max(1) < 0.5).sum()) if len(G) else len(A)
        else:
            s = -1
        if s > 0:
            scored.append((s, i))
    return [(t, i) for s, i in sorted(scored, reverse=True)[:k]]


thr_x = jload(run_file("six_xguard", "thresholds.json") or "_none_")
cfg_x, cfg_b = jload(run_file("six_xguard", "config.json") or "_none_"), jload(run_file("six_base", "config.json") or "_none_")
if thr_x and cfg_x and cfg_b:
    kx, kb = cfg_x["known"], cfg_b["known"]
    thr_b = jload(run_file("six_base", "thresholds.json"))
    # ---------- Fig. 3 qualitative known vs unknown ----------
    cases = []
    for t, kind, sub in [("sixray", "known_found", None), ("pidray", "known_found", None), ("pidray", "unknown_found", None),
                         ("pidray", "unknown_found", "hidden"), ("hixray", "unknown_found", None), ("opixray", "known_found", None)]:
        if t in AVAILABLE:
            cases += [(kind, sub) + c for c in pick("six_xguard", t, kind, kx, thr_x, sub)]
    if cases:
        fig, axes = plt.subplots(len(cases), 3, figsize=(11, 3.2 * len(cases)), squeeze=False)
        for row, (kind, sub, t, i) in zip(axes, cases):
            r = eval_set(t, True)[i]
            draw(row[0], r, None, kx, thr_x, "ground truth"); draw_gt(row[0], r)
            draw(row[1], r, preds_of("six_base", t)[i], kb, thr_b, "base")
            draw(row[2], r, preds_of("six_xguard", t)[i], kx, thr_x, "XGuard")
            row[0].set_ylabel(f"{t}{'/' + sub if sub else ''}\n{kind}", fontsize=8)
        fig.suptitle("Fig. 3  Known (green) and unknown (magenta) threat detection on real test images")
        plt.tight_layout(); savefig("fig3_qualitative")
    # ---------- Fig. 7 failure cases ----------
    fails = []
    for t in [x for x in ["sixray", "pidray", "hixray"] if x in AVAILABLE]:
        fails += [("missed",) + c for c in pick("six_xguard", t, "missed", kx, thr_x, k=1)]
        fails += [("false alarm",) + c for c in pick("six_xguard", t, "false_alarm", kx, thr_x, k=1)]
    if fails:
        fig, axes = plt.subplots(len(fails), 2, figsize=(8, 3.2 * len(fails)), squeeze=False)
        for row, (kind, t, i) in zip(axes, fails):
            r = eval_set(t, True)[i]
            draw(row[0], r, None, kx, thr_x, "ground truth"); draw_gt(row[0], r)
            draw(row[1], r, preds_of("six_xguard", t)[i], kx, thr_x, f"XGuard — {kind}")
            row[0].set_ylabel(t, fontsize=8)
        fig.suptitle("Fig. 7  Failure cases of XGuard (largest misses / false alarms)")
        plt.tight_layout(); savefig("fig7_failures")

# ---------- Fig. 4 occlusion robustness curves ----------
fig, ax = plt.subplots(figsize=(5, 3.6)); any_ = False
for t in ["six_base", "six_base-occaug", "six_oer", "six_sira+oer", "occ-cutout", "six_xguard"]:
    f = run_file(t, "metrics_occsweep.json")
    if f:
        d = jload(f); xs = sorted(float(k) for k in d)
        ax.plot([100 * x for x in xs], [100 * (d[str(x)] if d[str(x)] == d[str(x)] else np.nan) for x in xs], "o-", label=NICE.get(t, t)); any_ = True
if any_:
    ax.set_xlabel("synthetic occlusion of each object (%)"); ax.set_ylabel("AP50 (%)"); ax.legend(fontsize=7)
    ax.set_title("Fig. 4  Occlusion robustness (SIXray test, deterministic)")
    savefig("fig4_occlusion")
else:
    plt.close()

# ---------- Fig. 5 risk-coverage + referral ----------
tg5 = [t for t in ["pidray", "hixray", "sixray"] if run_file("six_xguard", f"curves_{t}.json")]
if tg5:
    fig, axes = plt.subplots(1, len(tg5) + 1, figsize=(3.6 * (len(tg5) + 1), 3.2), squeeze=False)
    grid = np.linspace(0.01, 1, 100)
    for ax, t in zip(axes[0], tg5):
        for tag, key, style, lab in [("six_base", "rc_msp", "--", "base · MSP"), ("six_xguard", "rc_msp", ":", "XGuard · MSP"),
                                     ("six_xguard", "rc_energy", "-.", "XGuard · energy"), ("six_xguard", "rc_rcr", "-", "XGuard · RCR")]:
            cv = jload(run_file(tag, f"curves_{t}.json") or "_none_", {}).get("all", {}).get(key)
            if cv:
                ax.plot(grid, cv, style, label=f"{lab} ({np.mean(cv) * 100:.1f})")
        ax.set_title(t); ax.set_xlabel("coverage"); ax.set_ylabel("selective risk"); ax.legend(fontsize=6)
    ax = axes[0][-1]
    for i, k in enumerate(RISK_TYPES):
        ax.bar(i, val("six_xguard", "pidray", f"miss_after_ref_{k}@10", fmt=False), color="#4C72B0")
    ax.set_xticks(range(len(RISK_TYPES))); ax.set_xticklabels(RISK_TYPES, rotation=30); ax.set_ylabel("missed threats after 10% referral (%)")
    ax.set_title("bag-level referral (PIDray)")
    fig.suptitle("Fig. 5  Risk–coverage (AURC in legend, lower is better) and operational referral", y=1.06)
    savefig("fig5_risk_referral")

# ---------- Fig. 6 CUS validity ----------
e = load_emb("six_xguard")
if e is not None and "synth" in e and len(e.get("synth", [])):
    known_x = (cfg_x or {}).get("known", [])
    Zn = e["z"] / np.linalg.norm(e["z"], axis=1, keepdims=True).clip(1e-6)
    mk = (e["ds"] == "sixray") & np.isin(e["cls"], known_x)
    mu = (e["ds"] == "pidray") & ~np.array([resolve_known(c, known_x) is not None for c in e["cls"]])
    pca = PCA(2).fit(np.concatenate([Zn[mk], e["synth"]]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, c in enumerate(sorted(set(e["cls"][mk]))):
        p = pca.transform(Zn[mk & (e["cls"] == c)]); axes[0].scatter(p[:, 0], p[:, 1], s=6, label=f"known {c}", alpha=0.6)
    p = pca.transform(e["synth"]); axes[0].scatter(p[:, 0], p[:, 1], s=8, marker="x", c="gray", label="CUS synthetic", alpha=0.5)
    if mu.any():
        p = pca.transform(Zn[mu]); axes[0].scatter(p[:, 0], p[:, 1], s=10, marker="^", c="crimson", label="REAL unseen (PIDray)", alpha=0.5)
    axes[0].legend(fontsize=6); axes[0].set_title("PCA of RoI embeddings")
    cen = np.stack([Zn[mk & (e["cls"] == c)].mean(0) for c in sorted(set(e["cls"][mk]))])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    for arr, lab in [(Zn[mk], "known (SIXray)"), (e["synth"], "CUS synthetic"), (Zn[mu] if mu.any() else None, "real unseen (PIDray)")]:
        if arr is not None and len(arr):
            axes[1].hist((arr @ cen.T).max(1), bins=40, alpha=0.5, density=True, label=lab)
    axes[1].set_xlabel("max cosine to a known-class centroid"); axes[1].legend(fontsize=7)
    axes[1].set_title("known-support overlap")
    fig.suptitle("Fig. 6  Are synthetic unknowns off the known support, like real unseen threats?")
    savefig("fig6_cus_validity")

# ---------- Fig. 8 per-class cross-domain AP heatmap ----------
rows = []
for tag in ["six_base", "six_xguard"]:
    for t in TG:
        f = run_file(tag, f"metrics_{t}.json")
        if f:
            for c, v in (jload(f)["all"].get("per_class_AP50") or {}).items():
                rows.append(dict(model=NICE.get(tag, tag), target=t, cls=c, AP50=100 * v if v == v else np.nan))
if rows:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, df.model.nunique(), figsize=(4.5 * df.model.nunique(), 3.6), squeeze=False)
    for ax, (mname, g) in zip(axes[0], df.groupby("model")):
        piv = g.pivot_table(index="cls", columns="target", values="AP50")
        im_ = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, "n/a" if v != v else f"{v:.0f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(mname, fontsize=9)
    fig.colorbar(im_, ax=axes[0].tolist(), shrink=0.8)
    fig.suptitle("Fig. 8  Per-class AP50 under cross-dataset shift (trained on SIXray only)")
    savefig("fig8_perclass_heatmap")
