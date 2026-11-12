# ==============================================================================
# 8. DATA PIPELINE + X-RAY AUGMENTATIONS
#    T_phi (scanner counterfactual): attenuation curve, pseudo-energy channel mixing,
#    colour-map rotation, saturation, contrast, detector saturation, blur, Poisson/Gaussian
#    noise -- geometry untouched, so boxes are unchanged.
#    Occlusion synthesis: x_occ = M*x + (1-M)*B  (replace) or Beer-Lambert overlay x*B.
#    Baselines: RGB jitter, Fourier amplitude mixing (FDA-style), Cutout.
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
def load_rgb(path, target_long):
    im = Image.open(path)
    try:
        im.draft("RGB", (target_long, target_long))
    except Exception:
        pass
    if im.mode not in ("RGB",):
        if im.mode in ("I;16", "I", "F"):
            a = np.asarray(im, dtype=np.float32); a = (a - a.min()) / max(1e-6, a.max() - a.min())
            im = Image.fromarray((a * 255).astype(np.uint8))
        im = im.convert("RGB")
    return im


class XrayDet(Dataset):
    def __init__(self, recs, cls2id, img_size, train=False, aug_seed=0):
        self.recs, self.cls2id, self.S, self.train, self.aug_seed = recs, cls2id, img_size, train, aug_seed

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        s = self.S / max(r["w"], r["h"])
        nw, nh = max(1, round(r["w"] * s)), max(1, round(r["h"] * s))
        try:
            im = load_rgb(r["path"], self.S).resize((nw, nh), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
        except Exception as e:
            print("image read error", r["path"], e)
            x = torch.zeros(3, nh, nw)
        keep = [j for j, c in enumerate(r["cls"]) if self.cls2id.get(c) is not None]
        boxes = torch.from_numpy(r["boxes"][keep] * s).float().reshape(-1, 4)
        labels = torch.tensor([self.cls2id[r["cls"][j]] for j in keep], dtype=torch.int64)
        if self.train and stable_hash(f"{self.aug_seed}-{i}") % 2:   # deterministic flip: exact resume
            x = x.flip(-1)
            if len(boxes):
                boxes = torch.stack([nw - boxes[:, 2], boxes[:, 1], nw - boxes[:, 0], boxes[:, 3]], 1)
        return x, dict(boxes=boxes, labels=labels, idx=i, scale=s)


def collate(b):
    return tuple(zip(*b))


class ResumableSampler(Sampler):
    """Deterministic per-epoch permutation that can start mid-epoch (for time-limited Kaggle sessions)."""
    def __init__(self, n, seed, epoch=0, start=0, weights=None):
        self.n, self.seed, self.epoch, self.start, self.weights = n, seed, epoch, start, weights

    def __iter__(self):
        g = np.random.RandomState(self.seed * 1000 + self.epoch)
        if self.weights is None:
            order = g.permutation(self.n)
        else:
            p = np.asarray(self.weights, np.float64); order = g.choice(self.n, self.n, replace=True, p=p / p.sum())
        return iter(order[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


# ---------------- GPU augmentations (batch lists of CHW tensors in [0,1]) ----------------
def _gauss_blur(x, sigma):
    if sigma < 0.3:
        return x
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(x[None], g.view(1, 1, 1, k).repeat(3, 1, 1, 1), padding=(0, k // 2), groups=3)
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), padding=(k // 2, 0), groups=3)
    return x[0]


def _hue_rotate(x, deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=x.device, dtype=x.dtype)
    R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=x.device, dtype=x.dtype)
    M = torch.linalg.inv(T) @ R @ T
    return torch.einsum("ij,jhw->ihw", M, x)


def scanner_counterfactual(x, strength=1.0):
    """T_phi: one random virtual scanner per image. x: CHW in [0,1]."""
    u = lambda a, b: a + (b - a) * random.random()
    s = strength
    x = x.clamp(0, 1) ** u(1 - 0.4 * s, 1 + 0.6 * s)                                   # attenuation curve
    M = torch.eye(3, device=x.device) + 0.15 * s * torch.randn(3, 3, device=x.device)   # pseudo dual-energy response
    M = M / M.sum(1, keepdim=True).clamp(min=0.3)
    x = torch.einsum("ij,jhw->ihw", M, x)
    x = _hue_rotate(x, u(-40, 40) * s)                                                  # colour mapping
    g = x.mean(0, keepdim=True); x = g + (x - g) * u(1 - 0.5 * s, 1 + 0.5 * s)          # saturation
    m = x.mean(); x = (x - m) * u(1 - 0.3 * s, 1 + 0.3 * s) + m + u(-0.08, 0.08) * s      # contrast / brightness
    sat = u(0.85, 1.0); x = x.clamp(0, sat) / sat                                       # detector saturation
    x = _gauss_blur(x, u(0, 1.5 * s))                                                   # focal-spot / sensor blur
    x = x + u(0, 0.04 * s) * torch.sqrt(x.clamp(min=0)) * torch.randn_like(x) + u(0, 0.01) * torch.randn_like(x)
    return x.clamp(0, 1)


def rgb_jitter(x):
    u = lambda a, b: a + (b - a) * random.random()
    g = x.mean(0, keepdim=True); x = g + (x - g) * u(0.7, 1.3)
    m = x.mean(); x = (x - m) * u(0.8, 1.2) + m + u(-0.1, 0.1)
    return _hue_rotate(x, u(-10, 10)).clamp(0, 1)


def fourier_mix(xs, beta=0.05):
    """FDA-style: swap low-frequency amplitude with another image of the batch."""
    out = []
    for i, x in enumerate(xs):
        y = xs[(i + 1) % len(xs)]
        y = F.interpolate(y[None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        fx, fy = torch.fft.fft2(x.float()), torch.fft.fft2(y.float())
        ax, px, ay = fx.abs(), fx.angle(), fy.abs()
        h, w = x.shape[-2:]; b = max(1, int(min(h, w) * beta))
        ax = torch.fft.fftshift(ax, dim=(-2, -1)); ay = torch.fft.fftshift(ay, dim=(-2, -1))
        ch, cw = h // 2, w // 2
        ax[:, ch - b:ch + b, cw - b:cw + b] = ay[:, ch - b:ch + b, cw - b:cw + b]
        ax = torch.fft.ifftshift(ax, dim=(-2, -1))
        out.append(torch.fft.ifft2(ax * torch.exp(1j * px)).real.clamp(0, 1).to(x.dtype))
    return out


def synth_occlusion(xs, targets, p=0.8, frac=(0.2, 0.6), mode="mixed", rng=None):
    """Occlude random parts of GT objects with clutter from other bags. Returns images + visibility maps (1=visible).
    rng: a random.Random for deterministic evaluation-time occlusion (occlusion-ratio sweep)."""
    rng = rng or random
    outs, vis = [], []
    for i, (x, t) in enumerate(zip(xs, targets)):
        x = x.clone(); v = torch.ones(1, *x.shape[-2:], device=x.device)
        src = xs[(i + 1) % len(xs)]
        H, W = x.shape[-2:]
        for b in t["boxes"]:
            if rng.random() > p:
                continue
            x1, y1, x2, y2 = [int(c) for c in b.tolist()]
            bw, bh = max(2, x2 - x1), max(2, y2 - y1)
            area = rng.uniform(*frac) * bw * bh
            ar = rng.uniform(0.3, 3.0)
            ow, oh = int(min(bw, max(2, math.sqrt(area * ar)))), int(min(bh, max(2, math.sqrt(area / ar))))
            ox, oy = rng.randint(x1, max(x1, x2 - ow)), rng.randint(y1, max(y1, y2 - oh))
            ox, oy = min(ox, W - ow), min(oy, H - oh)
            if ow < 2 or oh < 2 or ox < 0 or oy < 0:
                continue
            sh, sw = src.shape[-2:]
            sx, sy = rng.randint(0, max(0, sw - ow)), rng.randint(0, max(0, sh - oh))
            patch = src[:, sy:sy + oh, sx:sx + ow]
            if patch.shape[-2:] != (oh, ow):
                patch = F.interpolate(patch[None], size=(oh, ow), mode="bilinear", align_corners=False)[0]
            if mode == "cutout":
                x[:, oy:oy + oh, ox:ox + ow] = 0.5; v[:, oy:oy + oh, ox:ox + ow] = 0.0
            elif mode == "replace" or (mode == "mixed" and rng.random() < 0.5):
                x[:, oy:oy + oh, ox:ox + ow] = patch; v[:, oy:oy + oh, ox:ox + ow] = 0.0
            else:   # Beer-Lambert: overlapping objects multiply transmitted intensity
                dens = patch * rng.uniform(0.3, 0.9)
                x[:, oy:oy + oh, ox:ox + ow] = x[:, oy:oy + oh, ox:ox + ow] * dens
                v[:, oy:oy + oh, ox:ox + ow] = torch.minimum(v[:, oy:oy + oh, ox:ox + ow], dens.mean(0, keepdim=True))
        outs.append(x); vis.append(v)
    return outs, vis
