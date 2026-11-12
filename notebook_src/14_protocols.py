# ==============================================================================
# 13. PROTOCOLS + EXPERIMENT REGISTRY + RUNNER
#   P1 in-domain (SIXray)             P2 SIXray -> PIDray / OPIXray / HiXray (no target data, no target tuning)
#   P3 leave-one-domain-out           P4 leave-one-class-out on SIXray (folds = classes with enough objects)
#   P5 SIXray -> PIDray novel classes (never named)   P6 OPIXray OL1-3 (+ synthetic occlusion sweep)
#   P7 compound: PIDray-hidden (deliberately concealed) novel classes from a SIXray-only model
#   TIERS (budget): 1 = paper-critical, 2 = supporting ablations / baselines, 3 = extra studies
#   Every run folder holds fingerprint.json; a run whose training inputs changed is moved to _stale/ and
#   retrained, a run whose evaluation inputs changed is re-evaluated (weights kept). No stale caches.
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
AVAILABLE = set(REC)
RUNS_DIR = WORK / ("runs" if MODE in ("full", "report") else f"runs_{MODE}")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIRS = [RUNS_DIR]
if MODE == "report":     # + the other platform's finished runs
    for d in EXTRA_RESULT_DIRS:
        for pat in ["runs", "*/runs", "*/*/runs", "*/*/*/runs"]:
            RUNS_DIRS += [Path(x) for x in glob.glob(str(Path(d) / pat)) if Path(x) not in RUNS_DIRS]
    print("result folders merged:", [str(d) for d in RUNS_DIRS])


def all_runs():
    seen, out = set(), []
    for d in RUNS_DIRS:
        for rd in sorted(d.glob("*")):
            if rd.is_dir() and not rd.name.startswith("_") and rd.name not in seen:
                seen.add(rd.name); out.append(rd)
    return out


def run_path(name):
    for d in RUNS_DIRS:
        if (d / name / "DONE").exists():
            return d / name
    return RUNS_DIR / name
SIX_CLASSES = CLASS_AUDIT[(CLASS_AUDIT.dataset == "sixray")].cls.tolist() if len(CLASS_AUDIT) else []
LOCO_CLASSES = CLASS_AUDIT[(CLASS_AUDIT.dataset == "sixray") & CLASS_AUDIT.valid_loco].cls.tolist() if len(CLASS_AUDIT) else []
print("SIXray classes in the data:", SIX_CLASSES, "| valid LOCO folds:", LOCO_CLASSES)
if "hammer" not in SIX_CLASSES and "sixray" in AVAILABLE:
    print("⚠️  this SIXray mirror has NO 'hammer' -> it is a 5-class subset; do not present it as the canonical 6-class SIXray.")


def split_recs(ds, split):
    return [r for r in REC.get(ds, []) if r["split"] == split]


def cap(recs, n, salt=""):
    if n is None or len(recs) <= n:
        return recs
    return sorted(recs, key=lambda r: stable_hash(salt + r["uid"]))[:n]


def train_set(spec):
    out = []
    for ds in spec["train"]:
        rs = split_recs(ds, "train")
        if spec.get("loco"):
            rs = [r for r in rs if spec["loco"] not in r["cls"]]       # every image containing the held-out class is removed
        pos = sorted([r for r in rs if not r["neg"]], key=train_cap_order)
        pos = pos[: C.MAX_TRAIN_PER_DS] if C.MAX_TRAIN_PER_DS else pos
        out += pos + [r for r in rs if r["neg"]][: C.N_NEG_TRAIN]
    known = order_classes({c for r in out for c in r["cls"]})
    return out, known


def val_set(spec):
    rs = []
    n = C.TEST_VAL or C.MAX_VAL
    for ds in spec["train"]:
        v = [r for r in split_recs(ds, "val") if not r["neg"] and not r.get("excluded")]
        if spec.get("loco"):
            v = [r for r in v if spec["loco"] not in r["cls"]]
        rs += cap(v, max(1, n // len(spec["train"])), "val")
    return rs


EVAL_CACHE = {}


def eval_set(ds, full=True):
    """main runs: the complete test split; ablations: a FIXED md5-ordered subset (same images for every ablation)."""
    key = (ds, full)
    if key not in EVAL_CACHE:
        n = C.TEST_EVAL or (C.MAX_EVAL_MAIN if full else C.MAX_EVAL_ABL)
        t = split_recs(ds, "test")
        EVAL_CACHE[key] = cap([r for r in t if not r["neg"]], n, "ev") + [r for r in t if r["neg"]][: (C.TEST_EVAL or C.N_NEG_EVAL)]
    return EVAL_CACHE[key]


def S(name, group, tier, train, mcfg=None, kind="xguard", seed=0, targets=None, full_eval=False, epochs=None, **kw):
    return dict(name=f"{name}_s{seed}", tag=name, group=group, tier=tier, kind=kind, train=train, mcfg=mcfg or {}, seed=seed,
                targets=targets or sorted(AVAILABLE), full_eval=full_eval, epochs=epochs or (C.EPOCHS_MAIN if full_eval else C.EPOCHS_ABL), **kw)


def registry():
    R, six, X = [], ["sixray"], (lambda **k: variant("xguard", **k))
    for s in C.SEEDS_MAIN:      # ---------------- tier 1 (Colab A100)
        R += [S("six_base", "core", 1, six, variant("base"), seed=s, full_eval=True, analyze=True, occ_sweep=True),
              S("six_xguard", "core", 1, six, variant("xguard"), seed=s, full_eval=True, analyze=True, occ_sweep=True)]
    for s in C.SEEDS_ABL:
        for v in ["sira", "oer", "stpb", "sira+oer", "sira+oer+stpb", "sira+oer+stpb+cus", "xguard-sira", "xguard-oer", "xguard-cus"]:
            R.append(S(f"six_{v}", "ablation", 1, six, variant(v), seed=s, analyze=v == "sira",
                       occ_sweep=v in ("oer", "sira+oer")))
        for c in LOCO_CLASSES:
            for v in ["base", "xguard"]:
                R.append(S(f"loco-{c}_{v}", "loco", 2, six, variant(v), seed=s, loco=c,
                           targets=[t for t in ["sixray", "opixray", "pidray"] if t in AVAILABLE]))
        # ---------------- tier 2
        for src in ["v", "t", "a", "vt", "va"]:
            R.append(S(f"proto-{src}", "prototype", 2, six, X(stpb=src), seed=s))
        for g in ["none", "gaussian", "mixup", "slerp"]:
            R.append(S(f"unk-{g}", "unknown_gen", 2, six, X(unk_gen=g), seed=s))
        R.append(S("unk-cus-noent", "unknown_gen", 2, six, X(), seed=s, loss_w=dict(ent=0.0)))
        R += [S("scan-rgb", "sira_validation", 2, six, variant("base", aug="rgb"), seed=s, analyze=True),
              S("scan-fourier", "sira_validation", 2, six, variant("base", aug="fourier"), seed=s, analyze=True),
              S("occ-cutout", "occlusion", 2, six, variant("base", aug="cutout"), seed=s, occ_sweep=True)]
        R += [S("six_fcos", "baselines", 2, six, kind="fcos", seed=s, lr_mult=0.5, full_eval=True),
              S("six_retinanet", "baselines", 2, six, kind="retinanet", seed=s, lr_mult=0.5, full_eval=True),
              S("six_owlvit", "zeroshot", 2, six, kind="owlvit", seed=s), S("six_gdino", "zeroshot", 2, six, kind="gdino", seed=s),
              S("six_owlvit-oracle", "zeroshot", 2, six, kind="owlvit", oracle=True, seed=s),
              S("six_gdino-oracle", "zeroshot", 2, six, kind="gdino", oracle=True, seed=s)]
        # ---------------- tier 3
        R += [S("six_yolo11s", "ultralytics", 3, six, kind="yolo", weights="yolo11s.pt", seed=s, full_eval=True),
              S("six_rtdetr-l", "ultralytics", 3, six, kind="yolo", weights="rtdetr-l.pt", seed=s, full_eval=True)]
        for ds in ["opixray", "pidray", "hixray"]:
            for v in ["base", "xguard"]:
                R.append(S(f"in-{ds}_{v}", "indomain", 3, [ds], variant(v), seed=s, targets=[ds]))
        four = [d for d in ["sixray", "opixray", "pidray", "hixray"] if d in AVAILABLE]
        for held in four:
            for v in ["base", "xguard"]:
                R.append(S(f"lodo-{held}_{v}", "lodo", 3, [d for d in four if d != held], variant(v), seed=s, targets=[held]))
        R += [S("enc-imagenet", "encoder", 3, six, X(), seed=s, init="imagenet"), S("enc-dino", "encoder", 3, six, X(), seed=s, init="dino"),
              S("enc-clip", "encoder", 3, six, X(), seed=s, init="clip"), S("enc-frozen", "encoder", 3, six, X(), seed=s, tl=0),
              S("enc-unfrozen", "encoder", 3, six, X(), seed=s, tl=5),
              S("src2", "num_sources", 3, ["sixray", "opixray"], X(), seed=s, targets=[t for t in ["pidray", "hixray"] if t in AVAILABLE]),
              S("src3", "num_sources", 3, ["sixray", "opixray", "hixray"], X(), seed=s, targets=[t for t in ["pidray"] if t in AVAILABLE])]
    return R


REGISTRY = registry()
PRIORITY = ["core", "ablation", "loco", "sira_validation", "unknown_gen", "prototype", "occlusion", "baselines", "zeroshot",
            "indomain", "lodo", "num_sources", "encoder", "ultralytics"]
SELECTED = sorted([s for s in REGISTRY if s["tier"] in TIERS], key=lambda s: (PRIORITY.index(s["group"]), s["seed"], s["name"]))
if MODE == "test":   # one representative per configuration type of THIS platform's tiers exercises every code path
    keep_tags = {"six_base", "six_xguard", "six_sira", "six_oer", "six_sira+oer", "six_sira+oer+stpb+cus",       # tier 1
                 "proto-t", "proto-a", "unk-gaussian", "unk-slerp", "unk-cus-noent", "scan-fourier", "occ-cutout",  # tier 2
                 "six_fcos", "six_owlvit", "six_gdino-oracle", "in-opixray_xguard", "six_yolo11s", "src2"}          # tier 2-3
    keep_tags |= {f"loco-{LOCO_CLASSES[0]}_base", f"loco-{LOCO_CLASSES[0]}_xguard"} if LOCO_CLASSES else set()
    SELECTED = [s for s in SELECTED if s["tag"] in keep_tags and s["seed"] == 0]
if MODE == "report":
    SELECTED = []
print(f"{len(REGISTRY)} experiments registered; {len(SELECTED)} selected for MODE={MODE}, TIERS={TIERS}")
print(Counter(f"T{s['tier']}:{s['group']}" for s in SELECTED))


# ------------------------- fingerprints (stale-cache protection) -------------------------
def _src_hash(objs):
    parts = []
    for o in objs:
        try:
            parts.append(inspect.getsource(o))
        except Exception:
            parts.append(getattr(o, "__name__", str(o)))
    return sha(parts)


def fingerprints(spec, train_recs, val_recs, known):
    TRAIN_CODE = _src_hash([XGuard, STPB, CUS, OER, SIRAAdapter, different_class_pairs, train_model, prepare_views,
                            scanner_counterfactual, synth_occlusion, XrayDet, build_backbone_rpn, TVDetector])
    EVAL_CODE = _src_hash([evaluate, compute_thresholds, build_eval_items, coco_eval, infer, greedy_match, risk_coverage])
    cfg = {k: getattr(C, k) for k in ["IMG", "BATCH", "ACCUM", "LR", "WARMUP_ITERS", "WEIGHT_DECAY", "TEST_ITERS", "OCC_P", "OCC_FRAC",
                                     "VIEW2_DET", "LOSS_W", "EMB_DIM", "AMP", "GRAD_CLIP", "MAX_TRAIN_PER_DS", "PACK_LONG"]}
    train_fp = sha(dict(code=CODE_VERSION, src=TRAIN_CODE, spec={k: v for k, v in spec.items() if k not in ("targets", "full_eval", "analyze", "occ_sweep")},
                        cfg=cfg, train=sha(sorted(r["uid"] for r in train_recs)), val=sha(sorted(r["uid"] for r in val_recs)),
                        known=known, ontology=ONTOLOGY_VERSION, amp=str(AMP_DTYPE)))
    evals = {t: sha(sorted(r["uid"] for r in eval_set(t, spec["full_eval"]))) for t in spec["targets"] if t in AVAILABLE}
    eval_fp = sha(dict(src=EVAL_CODE, evals=evals, thr=[C.REFERRAL_BUDGETS, C.SCORE_THR, C.DETS_PER_IMG], occ=C.OCC_SWEEP,
                       alias=C.USE_SEMANTIC_ALIASES, ontology=ONTOLOGY_VERSION, train_fp=train_fp))
    return dict(train_fp=train_fp, eval_fp=eval_fp, train_code=TRAIN_CODE, eval_code=EVAL_CODE)


def check_stale(rd, fp):
    old = jload(rd / "fingerprint.json")
    if old is None:
        return
    if old["train_fp"] != fp["train_fp"]:
        dst = RUNS_DIR / "_stale" / f"{rd.name}_{time.strftime('%Y%m%d-%H%M%S')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rd), str(dst))
        print(f"   ♻️  training inputs/code changed -> old run moved to {dst} and retrained")
    elif old["eval_fp"] != fp["eval_fp"]:
        for f in list(rd.glob("preds_*.pkl")) + list(rd.glob("metrics_*.json")) + list(rd.glob("curves_*.json")) + \
                [rd / "thresholds.json", rd / "DONE", rd / "emb.npz"]:
            if f.exists():
                f.unlink()
        print("   ♻️  evaluation code/sets changed -> predictions and metrics recomputed (weights kept)")


# ------------------------- zero-shot & ultralytics adapters -------------------------
class ZeroShotOV(nn.Module):
    """OWL-ViT / Grounding-DINO. Name-free protocol: known-class prompts + ONE generic prompt for unknowns.
    oracle=True: the unseen class NAMES are also given (vocabulary disclosure) -> an upper-bound REFERENCE only."""
    GENERIC = "a dangerous object"

    def __init__(self, kind, known, oracle_names=None):
        super().__init__()
        from transformers import AutoProcessor
        self.kind, self.known, self.m = kind, list(known), dict(kind=kind, oracle=bool(oracle_names))
        name = "google/owlvit-base-patch32" if kind == "owlvit" else "IDEA-Research/grounding-dino-tiny"
        self.proc = AutoProcessor.from_pretrained(name)
        if kind == "owlvit":
            from transformers import OwlViTForObjectDetection as M
        else:
            from transformers import AutoModelForZeroShotObjectDetection as M
        self.net = M.from_pretrained(name).to(DEVICE).eval()
        self.init_used = name
        extra = [TEXT_NAME.get(c, c.replace("_", " ")) for c in (oracle_names or [])] or [self.GENERIC]
        self.queries = [TEXT_NAME.get(c, c.replace("_", " ")) for c in self.known] + extra

    @torch.no_grad()
    def forward(self, xs):
        res = []
        for x in xs:
            im = Image.fromarray((x.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8))
            H, W = x.shape[-2:]
            if self.kind == "owlvit":
                inp = self.proc(text=[self.queries], images=im, return_tensors="pt").to(DEVICE)
                out = self.net(**inp)
                r = self.proc.post_process_object_detection(out, threshold=C.SCORE_THR, target_sizes=torch.tensor([[H, W]], device=DEVICE))[0]
                b, s, l = r["boxes"].float().cpu(), r["scores"].float().cpu(), r["labels"].cpu()
            else:
                inp = self.proc(images=im, text=". ".join(self.queries) + ".", return_tensors="pt").to(DEVICE)
                out = self.net(**inp)
                try:
                    r = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=[(H, W)])[0]
                except TypeError:
                    r = self.proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.05, text_threshold=0.05, target_sizes=[(H, W)])[0]
                b, s = r["boxes"].float().cpu(), r["scores"].float().cpu()
                txt = r.get("text_labels", r.get("labels"))
                l = torch.tensor([next((i for i, q in enumerate(self.queries) if q in str(t)), len(self.queries) - 1) for t in txt], dtype=torch.long)
            unk = l >= len(self.known)
            kb, ks, kl = b[~unk], s[~unk], l[~unk]
            nan_k, nan_u = torch.full_like(ks, float("nan")), torch.full_like(s[unk], float("nan"))
            res.append(dict(boxes=kb, scores=ks, labels=kl, U=1 - ks, R=1 - ks, E=nan_k, H=nan_k,
                            ub=b[unk], us=s[unk], uR=1 - s[unk], uE=nan_u, uH=nan_u))
        return res


def run_ultralytics(spec, train_recs, known, rd, epochs):
    if not INTERNET:
        raise RuntimeError("ultralytics baseline needs Internet (pip + weights)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
    from ultralytics import YOLO, RTDETR
    root = CACHE / "yolo_export" / "_".join(spec["train"])
    cls2id = {c: i for i, c in enumerate(known)}
    for split, recs in [("train", train_recs), ("val", val_set(spec))]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True); (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in recs:
            name = r["uid"].replace("/", "_") + os.path.splitext(r["path"])[1]
            dst = root / "images" / split / name
            if not dst.exists():
                os.symlink(r["path"], dst)
            with open(root / "labels" / split / (os.path.splitext(name)[0] + ".txt"), "w") as f:
                for b, c in zip(r["boxes"], r["cls"]):
                    if c in cls2id:
                        x1, y1, x2, y2 = b
                        f.write(f"{cls2id[c]} {(x1 + x2) / 2 / r['w']:.6f} {(y1 + y2) / 2 / r['h']:.6f} {(x2 - x1) / r['w']:.6f} {(y2 - y1) / r['h']:.6f}\n")
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: {json.dumps(known)}\n")
    M_ = RTDETR if "rtdetr" in spec["weights"] else YOLO
    model = M_(spec["weights"])
    model.train(data=str(root / "data.yaml"), epochs=epochs, imgsz=C.IMG, batch=C.BATCH * 2, seed=spec["seed"], project=str(rd),
                name="ul", exist_ok=True, verbose=False, plots=False, fraction=0.05 if C.TEST_ITERS else 1.0)

    class Wrap(nn.Module):
        def __init__(s):
            super().__init__(); s.init_used = spec["weights"]; s.m = dict(kind="ultralytics")

        @torch.no_grad()
        def forward(s, xs):
            ims = [(x.permute(1, 2, 0).float().cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8) for x in xs]
            out = []
            for r in model.predict(ims, imgsz=C.IMG, conf=C.SCORE_THR, verbose=False, max_det=C.DETS_PER_IMG):
                b = torch.from_numpy(r.boxes.xyxy.cpu().numpy()); sc = torch.from_numpy(r.boxes.conf.cpu().numpy())
                l = torch.from_numpy(r.boxes.cls.cpu().numpy()).long(); nan = torch.full_like(sc, float("nan"))
                out.append(dict(boxes=b, scores=sc, labels=l, U=1 - sc, R=1 - sc, E=nan, H=nan, ub=b, us=1 - sc, uR=1 - sc, uE=nan, uH=nan))
            return out
    return Wrap()


# ------------------------- runner -------------------------
@torch.no_grad()
def extract_embeddings(model, n_img=150):
    """GT-object embeddings of every test set (+ scanner-counterfactual consistency on SIXray, CUS samples)."""
    if not hasattr(model, "embed_boxes"):
        return None
    Z, D, Y, SYN, CONS = [], [], [], None, []
    allc = {c: i for i, c in enumerate(GLOBAL_ORDER + sorted({c for r in ALL_REC for c in r["cls"]} - set(GLOBAL_ORDER)))}
    inv = {i: c for c, i in allc.items()}
    for ds in sorted(AVAILABLE):
        recs = cap([r for r in split_recs(ds, "test") if not r["neg"] and not r.get("excluded")], n_img, "emb")
        for bi, (xs, ts) in enumerate(make_loader(recs, allc, False, bs=4)):
            xs = [x.to(DEVICE) for x in xs]
            bx = [t["boxes"].to(DEVICE) for t in ts]
            if sum(len(b) for b in bx) == 0:
                continue
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=C.AMP):
                z = model.embed_boxes(xs, bx)
                if ds == "sixray":   # SIRA claim: same object under a different virtual scanner -> same embedding
                    random.seed(1000 + bi)
                    z2 = model.embed_boxes([scanner_counterfactual(x) for x in xs], bx)
                    CONS += F.cosine_similarity(z.float(), z2.float(), dim=-1).cpu().tolist()
            Z.append(z.float().cpu().numpy()); D += [ds] * len(z); Y += [inv[int(l)] for t in ts for l in t["labels"]]
    if not Z:
        return None
    Z = np.concatenate(Z)
    if getattr(model, "cus", None) is not None:
        known = model.known
        m = np.array([y in known for y in Y]) & (np.array(D) == "sixray")
        if m.sum() > 4:
            zr = torch.from_numpy(Z[m]).to(DEVICE)
            yr = torch.tensor([known.index(y) for y, k in zip(Y, m) if k], device=DEVICE)
            out = model.cus.synth(zr, yr, 400)
            if out is not None:
                SYN = F.normalize(out.float(), dim=-1).cpu().numpy()
    return dict(z=Z, ds=np.array(D), cls=np.array(Y), synth=SYN, scanner_cos=np.array(CONS))


def write_run_artifacts(rd, spec, fp, known):
    jdump(fp, rd / "fingerprint.json")
    jdump(dict(spec=spec, C={k: v for k, v in vars(C).items()}, MODE=MODE, known=known), rd / "config.json")
    (rd / "dataset_manifest_hash.txt").write_text(str(MANIFEST_HASH))
    (rd / "ontology_version.txt").write_text(ONTOLOGY_VERSION)
    (rd / "git_commit.txt").write_text(ENV_INFO.get("git_commit", "n/a"))
    (rd / "seed.txt").write_text(str(spec["seed"]))
    jdump(ENV_INFO, rd / "environment.json")


def load_preds(pf, recs):
    d, sp = pickle.load(open(pf, "rb"))
    return [d.get(r["uid"]) for r in recs], sp


def run_experiment(spec):
    rd = RUNS_DIR / spec["name"]
    miss = [d for d in spec["train"] if d not in AVAILABLE]
    targets = [t for t in spec["targets"] if t in AVAILABLE]
    if miss or not targets:
        return f"skipped (missing datasets {miss or spec['targets']})"
    if spec["kind"] in ("owlvit", "gdino", "yolo") and not INTERNET:
        return "skipped (needs Internet for weights)"
    train_recs, known = train_set(spec)
    vrecs = val_set(spec)
    fp = fingerprints(spec, train_recs, vrecs, known)
    check_stale(rd, fp)
    if (rd / "DONE").exists():
        return "done (fingerprint verified)"
    rd.mkdir(parents=True, exist_ok=True)
    write_run_artifacts(rd, spec, fp, known)
    t0 = time.time()
    print(f"\n▶ {spec['name']} [T{spec['tier']} {spec['group']}] train={spec['train']} images={len(train_recs)} known={known} epochs={spec['epochs']}")
    if spec["kind"] in ("owlvit", "gdino"):
        oracle = sorted({c for t in targets for r in eval_set(t, spec["full_eval"]) for c in r["cls"]
                         if resolve_known(c, known) is None}) if spec.get("oracle") else None
        model, meta = ZeroShotOV(spec["kind"], known, oracle), dict(init_used="zero-shot", oracle_vocabulary=oracle)
        jdump(meta, rd / "train_meta.json")
    elif spec["kind"] == "yolo":
        model, meta = run_ultralytics(spec, train_recs, known, rd, 1 if C.TEST_ITERS else spec["epochs"]), dict(init_used=spec["weights"])
        jdump(meta, rd / "train_meta.json")
    else:
        model, meta = train_model(spec, rd, train_recs, known, vrecs, spec["epochs"])
    thr = jload(rd / "thresholds.json")
    if thr is None:
        vp, _ = infer(model, vrecs)
        thr = compute_thresholds(vp, vrecs, known); jdump(thr, rd / "thresholds.json")
    dep = dict(train_minutes=meta.get("train_minutes"), sec_per_iter=meta.get("sec_per_iter"), init_used=meta.get("init_used"),
               gpu=GPU_NAME, platform=PLATFORM, amp=str(AMP_DTYPE), batch=C.BATCH, accum=C.ACCUM)
    eval_minutes = 0.0
    for tgt in targets:
        recs = eval_set(tgt, spec["full_eval"])
        pf = rd / f"preds_{tgt}.pkl"
        wait_pack(tgt)
        te = time.time()
        if pf.exists():
            preds, sp = load_preds(pf, recs)
        else:
            preds, sp = infer(model, recs)
            pickle.dump(({r["uid"]: p for r, p in zip(recs, preds)}, sp), open(pf, "wb"))
        eval_minutes += (time.time() - te) / 60
        dep.setdefault("fps", sp["fps"]); dep.setdefault("ms_per_img", sp["ms_per_img"]); dep.setdefault("eval_images", 0)
        dep["eval_images"] += len(recs)
        subsets = ["all"] + sorted({s for r in recs for s in r["subsets"]})
        out, curves = {}, {}
        for sub in subsets:
            out[sub], cv = evaluate(preds, recs, known, thr, None if sub == "all" else sub)
            curves[sub] = {k: (v.tolist() if v is not None else None) for k, v in cv.items()}
        jdump(out, rd / f"metrics_{tgt}.json"); jdump(curves, rd / f"curves_{tgt}.json")
        a = out["all"]
        print(f"   {tgt:8s} AP50={a.get('AP50', np.nan):.3f} xAP50={a.get('xAP50', np.nan):.3f} hAP50={a.get('hAP50', np.nan):.3f} "
              f"U-AP50={a.get('U_AP50', np.nan):.3f} U-R={a.get('U_R50', np.nan):.3f} locAUROC={a.get('locAUROC', np.nan):.3f} "
              f"AURC msp/rcr={a.get('AURC_msp', np.nan):.3f}/{a.get('AURC_rcr', np.nan):.3f}")
        bad = {k: v for k, v in a.get("metric_status", {}).items() if v.startswith("evaluator_failure")}
        if bad:
            print("   ❌ evaluator failures:", bad)
    if spec.get("occ_sweep") and "sixray" in targets and spec["kind"] == "xguard":
        recs = [r for r in eval_set("sixray", False) if not r["neg"]]
        sweep = {}
        for r_ in C.OCC_SWEEP:
            p_, _ = infer(model, recs, occ_ratio=r_)
            mm, _ = evaluate(p_, recs, known, thr, light=True)
            sweep[str(r_)] = mm.get("AP50")
        jdump(sweep, rd / "metrics_occsweep.json")
        print("   occlusion sweep AP50:", {k: round(v, 3) if v == v else v for k, v in sweep.items()})
    dep["eval_minutes"] = eval_minutes
    if isinstance(model, nn.Module) and spec["kind"] not in ("owlvit", "gdino", "yolo"):
        dep["params_M"] = count_params(model); dep["GFLOPs"] = model_flops(model)
    jdump(dep, rd / "deploy.json")
    if spec.get("analyze") and not (rd / "emb.npz").exists():
        wait_pack_all()
        e = extract_embeddings(model, 40 if C.TEST_ITERS else 150)
        if e is not None:
            np.savez_compressed(rd / "emb.npz", **{k: v for k, v in e.items() if v is not None})
    (rd / "DONE").write_text(f"{(time.time() - t0) / 60:.1f} min")
    del model; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return "finished"
