# ==============================================================================
# 10. TRAINING + INFERENCE (reproducible & resumable)
#   * full checkpoint (model, optimizer, scaler, epoch, step, python/numpy/torch/cuda RNG) -> exact resume,
#     the deterministic per-epoch sampler restarts mid-epoch at the saved position
#   * gradient accumulation: the incomplete last window of an epoch is applied, not dropped
#   * non-finite losses are counted; the run FAILS if they exceed C.MAX_NAN_FRAC (no silent recovery)
#   * the checkpoint is selected on the SOURCE validation split (harmonized AP50); never on target data
#   * artifacts: checkpoint_best.pt, checkpoint_last.pt (fp16 weights), train_log.csv, validation_metrics.json
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
class TimeUp(Exception):
    pass


_FORCE_TIMEUP_AT = None   # used only by the resume test


def loss_weight(k, m, override=None):
    W = dict(C.LOSS_W); W.update(override or {})
    table = dict(sira=W["sira"], cons_obj=W["occ"] if m.get("occ") else W["sira"], rel=W["rel"], proto=W["proto"],
                 threat=W["threat"], unknown=W["unknown"], ent=W["ent"], cons=W["cons"], risk=W["risk"], rec=W["rec"])
    return table.get(k, 1.0)


def prepare_views(xs, ts, m):
    aug = m.get("aug", "none")
    if aug == "rgb":
        xs = [rgb_jitter(x) for x in xs]
    elif aug == "fourier" and len(xs) > 1:
        xs = [a if random.random() < 0.5 else b for a, b in zip(xs, fourier_mix(xs))]
    elif aug == "cutout":
        xs, _ = synth_occlusion(xs, ts, p=C.OCC_P, frac=C.OCC_FRAC, mode="cutout")
    if not (m.get("sira_cf") or m.get("occ")):
        return xs, None, None
    x2 = [scanner_counterfactual(x) for x in xs] if m.get("sira_cf") else [x.clone() for x in xs]
    vis = None
    if m.get("occ"):
        x2, vis = synth_occlusion(x2, ts, p=C.OCC_P, frac=C.OCC_FRAC, mode="mixed")
    return xs, x2, vis


def make_loader(recs, cls2id, train, seed=0, epoch=0, start=0, bs=None):
    ds = XrayDet(recs, cls2id, C.IMG, train, aug_seed=seed * 1000 + epoch)
    sampler = ResumableSampler(len(ds), seed, epoch, start) if train else None
    kw = dict(persistent_workers=False, prefetch_factor=4) if C.WORKERS > 0 else {}
    return DataLoader(ds, batch_size=bs or C.BATCH, sampler=sampler, shuffle=False, num_workers=C.WORKERS,
                      collate_fn=collate, pin_memory=DEVICE.type == "cuda", drop_last=False, **kw)


def save_weights(model, path):
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}, path)


def load_weights(spec, known, path, meta):
    sp = dict(spec); sp["init"] = "clip" if spec.get("init") == "clip" else "none"
    global INTERNET, FORCE_NO_TEXT
    net, INTERNET = INTERNET, (INTERNET if sp["init"] == "clip" else False)
    FORCE_NO_TEXT = meta.get("text_available") is False
    try:
        model = build_model(sp, known)
    finally:
        INTERNET, FORCE_NO_TEXT = net, False
    sd = torch.load(path, map_location="cpu")
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "num_batches_tracked" not in k]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint/architecture mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")
    return model.to(DEVICE).eval()


def validate(model, val_recs, known):
    """Source-validation score for checkpoint selection: harmonized AP50 (target data is never used)."""
    if not val_recs:
        return float("nan")
    p, _ = infer(model, val_recs)
    m, _ = evaluate(p, val_recs, known, dict(tau_known=0.5, tau_U=0.5, tau_R=0.5, bag=None), light=True)
    model.train()
    return float(m.get("AP50", float("nan")))


def train_model(spec, run_dir, train_recs, known, val_recs, epochs):
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_p = run_dir / "train_meta.json"
    if (run_dir / "checkpoint_best.pt").exists() and meta_p.exists():
        meta = jload(meta_p)
        return load_weights(spec, known, run_dir / "checkpoint_best.pt", meta), meta
    seed = spec.get("seed", 0); seed_everything(seed)
    cls2id = {c: i for i, c in enumerate(known)}
    model = build_model(spec, known).to(DEVICE)
    m = spec.get("mcfg", {})
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = C.LR * (C.BATCH * C.ACCUM) / 8 * spec.get("lr_mult", 1.0)
    opt = torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)
    ipe = math.ceil(len(train_recs) / C.BATCH)
    if C.TEST_ITERS:
        ipe, epochs = min(ipe, C.TEST_ITERS), 1
    total = max(1, epochs * math.ceil(ipe / C.ACCUM))
    warm = min(C.WARMUP_ITERS, max(1, total // 5))
    ep0, it0, gstep, log, vlog, best, nan_total, steps_total = 0, 0, 0, [], [], -1.0, 0, 0
    ck = run_dir / "checkpoint_last_full.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        ep0, it0, gstep, log, vlog, best = s["epoch"], s["it"], s["gstep"], s["log"], s["vlog"], s["best"]
        nan_total, steps_total = s["nan_total"], s["steps_total"]
        set_rng_state(s["rng"])
        print(f"   resumed from epoch {ep0} iter {it0} (best val AP50 so far {best:.4f})")

    def lr_at(g):
        if g < warm:
            return base_lr * (0.001 + 0.999 * g / warm)
        p = (g - warm) / max(1, total - warm)
        return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    def checkpoint(ep, it):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep, it=it,
                        gstep=gstep, log=log, vlog=vlog, best=best, nan_total=nan_total, steps_total=steps_total,
                        rng=rng_state()), ck)

    def step():
        nonlocal gstep
        for g in opt.param_groups:
            g["lr"] = lr_at(gstep)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, C.GRAD_CLIP)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); gstep += 1

    t0 = time.time()
    model.train()
    lw = spec.get("loss_w")
    for ep in range(ep0, epochs):
        loader = make_loader(train_recs, cls2id, True, seed, ep, it0 * C.BATCH)
        _rs = torch.get_rng_state(); batches = iter(loader); torch.set_rng_state(_rs)   # loader must not consume the training RNG
        run_l, pending = defaultdict(float), 0
        last_bi = it0 - 1
        for bi, (xs, ts) in enumerate(batches, start=it0):
            if bi >= ipe:
                break
            last_bi = bi
            xs = [x.to(DEVICE, non_blocking=True) for x in xs]
            ts = [dict(boxes=t["boxes"].to(DEVICE), labels=t["labels"].to(DEVICE)) for t in ts]
            x1, x2, vis = prepare_views(xs, ts, m)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=C.AMP):
                L = model.forward_train(x1, ts, xs2=x2, vis2=vis)
            loss = sum(loss_weight(k, m, lw) * v.float() for k, v in L.items() if torch.is_tensor(v))
            steps_total += 1
            if not torch.isfinite(loss):
                nan_total += 1; opt.zero_grad(set_to_none=True); pending = 0
                if nan_total > max(20, C.MAX_NAN_FRAC * steps_total):
                    raise RuntimeError(f"{nan_total}/{steps_total} non-finite training steps -> run FAILED (not reported)")
                continue
            scaler.scale(loss / C.ACCUM).backward(); pending += 1
            if pending == C.ACCUM:
                step(); pending = 0
            for k, v in L.items():
                if torch.is_tensor(v):
                    run_l[k] += float(v)
            run_l["total"] += float(loss); run_l["_n"] += 1
            if (bi + 1) % 50 == 0 or bi + 1 == ipe:
                n = max(1, run_l.pop("_n"))
                msg = {k: round(v / n, 4) for k, v in run_l.items()}
                diag = {k: round(float(np.mean(v)), 4) for k, v in getattr(model, "diag", {}).items() if v}
                if hasattr(model, "diag"):
                    model.diag.clear()
                log.append(dict(epoch=ep, it=bi + 1, lr=opt.param_groups[0]["lr"], nan_total=nan_total,
                                unk_skipped=getattr(model, "unk_skipped", 0), **msg, **diag))
                print(f"   ep {ep + 1}/{epochs} it {bi + 1}/{ipe} lr {opt.param_groups[0]['lr']:.4f} " +
                      " ".join(f"{k}={v:.3f}" for k, v in msg.items()) +
                      (" | " + " ".join(f"{k}={v:.3f}" for k, v in diag.items()) if diag else "") +
                      f"  [{(time.time() - t0) / 60:.1f} min]")
                run_l = defaultdict(float)
            if time_left_h() < 0.3 or (_FORCE_TIMEUP_AT is not None and steps_total == _FORCE_TIMEUP_AT):
                if pending:
                    step(); pending = 0
                checkpoint(ep, bi + 1); raise TimeUp(f"session time nearly over: checkpoint saved at epoch {ep} it {bi + 1}")
        if pending:                       # apply the incomplete accumulation window
            step(); pending = 0
        it0 = 0
        if (ep + 1) % C.VAL_EVERY == 0 or ep + 1 == epochs:
            v = validate(model, val_recs, known)
            vlog.append(dict(epoch=ep + 1, val_AP50=v, minutes=(time.time() - t0) / 60))
            print(f"   epoch {ep + 1}: source-val AP50 = {v:.4f}" + ("  (best)" if v > best else ""))
            if v > best or math.isnan(best):
                best = v; save_weights(model, run_dir / "checkpoint_best.pt")
        save_weights(model, run_dir / "checkpoint_last.pt")
        checkpoint(ep + 1, 0)
    if not (run_dir / "checkpoint_best.pt").exists():
        save_weights(model, run_dir / "checkpoint_best.pt")
    pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
    jdump(vlog, run_dir / "validation_metrics.json")
    meta = dict(init_used=getattr(model, "init_used", None), params_M=count_params(model), train_images=len(train_recs),
                train_minutes=(time.time() - t0) / 60, epochs=epochs, iters_per_epoch=ipe, img=C.IMG, spec=spec,
                text_available=getattr(getattr(model, "stpb", None), "text_available", None),
                prototype_sources=model.stpb.sources() if hasattr(model, "stpb") else None,
                prototype_mix=model.stpb.mix_weights() if hasattr(model, "stpb") else {},
                text_info=dict(TEXT_INFO), nan_steps=nan_total, steps=steps_total,
                unk_skipped=getattr(model, "unk_skipped", 0), best_val_AP50=best,
                sec_per_iter=(time.time() - t0) / max(1, steps_total - 0))
    jdump(meta, meta_p)
    if not C.KEEP_FULL_CKPT and ck.exists():
        ck.unlink()
    return load_weights(spec, known, run_dir / "checkpoint_best.pt", meta), meta


@torch.no_grad()
def infer(model, recs, bs=None, occ_ratio=0.0):
    """occ_ratio > 0: deterministic synthetic occlusion of every GT object (evaluation-time occlusion sweep)."""
    model.eval()
    allc = {c: i for i, c in enumerate(sorted({c for r in recs for c in r["cls"]}))} if occ_ratio > 0 else {}
    loader = make_loader(recs, allc, False, bs=bs or max(1, C.BATCH))
    preds, t_model, n = [None] * len(recs), 0.0, 0
    for bi, (xs, ts) in enumerate(loader):
        xs = [x.to(DEVICE, non_blocking=True) for x in xs]
        if occ_ratio > 0:
            xs, _ = synth_occlusion(xs, [dict(boxes=t["boxes"].to(DEVICE)) for t in ts], p=1.0, frac=(occ_ratio, occ_ratio),
                                    mode="mixed", rng=random.Random(7919 * (bi + 1)))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=C.AMP):
            outs = model(xs)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_model += time.time() - t; n += len(xs)
        for o, tg in zip(outs, ts):
            s = tg["scale"]
            preds[tg["idx"]] = {k: (v.float().numpy() / (s if k in ("boxes", "ub") else 1.0)).astype(np.float32)
                                if k != "labels" else v.numpy().astype(np.int64) for k, v in o.items()}
    return preds, dict(ms_per_img=1000 * t_model / max(1, n), fps=n / max(1e-6, t_model))


def model_flops(model):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = [torch.rand(3, C.IMG, C.IMG, device=DEVICE)]
        fc = FlopCounterMode(display=False)
        with fc, torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=C.AMP):
            model(x)
        return fc.get_total_flops() / 2e9   # GMACs-style convention (FLOPs/2)
    except Exception:
        return float("nan")
