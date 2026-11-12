# ==============================================================================
# 9. XGUARD MODEL
#    Two-stage detector (ResNet-50-FPN + RPN + RoI head).  NOTE: the proposal names a
#    DINO/Deformable-DETR decoder; on a single Kaggle T4/P100 a DETR-family model does not
#    converge within the session budget, so the five modules are implemented as
#    detector-agnostic plug-ins on RoI (query-equivalent) embeddings. State this in the paper.
#
#    SIRA : Scanner-Invariant Representation Adapter (frequency-conditioned gate on FPN
#           features + counterfactual-scanner consistency)
#    OER  : Occlusion Evidence Router (token reliability r_i, attention bias lambda*log r_i,
#           synthetic evidence removal + consistency + visibility supervision of r_i)
#    STPB : Semantic Threat Prototype Bank (visual + CLIP-text + attribute prototypes,
#           superclass hierarchy, generic threat prototype)
#    CUS  : Counterfactual Unknown Synthesis (shape/material/function factor mixing)
#    RCR  : Risk-Calibrated Referral head  R_i = P(detection incorrect | z_i)
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import MultiScaleRoIAlign, box_iou, batched_nms, nms, roi_align
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.image_list import ImageList

DINO_R50_URL = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def stage_local_weights():
    """If pretrained .pth files were attached as a Kaggle dataset, put them where torch.hub looks (offline use)."""
    dst = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"; dst.mkdir(parents=True, exist_ok=True)
    if PLATFORM != "kaggle":
        return
    for p in glob.glob(os.path.join(INPUT_ROOT, "**", "*.pth"), recursive=True)[:50]:
        n = os.path.basename(p)
        if any(s in n for s in ["fasterrcnn", "resnet50", "dino_resnet50", "fcos", "retinanet"]) and not (dst / n).exists():
            try:
                os.symlink(p, dst / n)
            except Exception:
                shutil.copy(p, dst / n)


stage_local_weights()


def default_rpn():
    anchor = AnchorGenerator(((32,), (64,), (128,), (256,), (512,)), ((0.5, 1.0, 2.0),) * 5)
    return RegionProposalNetwork(anchor, RPNHead(256, 3), 0.7, 0.3, 256, 0.5, dict(training=2000, testing=1000),
                                 dict(training=2000, testing=1000), 0.7)


class ClipRN50FPN(nn.Module):
    """CLIP-family encoder (open_clip RN50 visual tower) + FPN."""
    def __init__(self, tl):
        super().__init__()
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
        self.v = m.visual.float()
        del self.v.attnpool
        layers = ["layer4", "layer3", "layer2", "layer1"][:tl]
        for n, p in self.v.named_parameters():
            p.requires_grad_(any(n.startswith(l) for l in layers))
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, extra_blocks=LastLevelMaxPool())
        self.out_channels = 256

    def forward(self, x):
        v = self.v
        if hasattr(v, "stem"):
            x = v.stem(x)
        else:
            x = v.act1(v.bn1(v.conv1(x))); x = v.act2(v.bn2(v.conv2(x))); x = v.act3(v.bn3(v.conv3(x))); x = v.avgpool(x)
        c2 = v.layer1(x); c3 = v.layer2(c2); c4 = v.layer3(c3); c5 = v.layer4(c4)
        return self.fpn(OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)]))


def build_backbone_rpn(init, tl):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models import resnet50, ResNet50_Weights
    order = [init] + [x for x in ["imagenet", "none"] if x != init]
    for ini in order:
        if ini != "none" and not INTERNET and not any(Path(os.environ["TORCH_HOME"]).rglob("*.pth")):
            print(f"[init] no internet and no local weights -> skip '{ini}'"); continue
        try:
            if ini == "coco":
                m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, trainable_backbone_layers=tl)
                return m.backbone, m.rpn, ini
            if ini == "clip":
                return ClipRN50FPN(tl), default_rpn(), ini
            r = resnet50(weights=None, norm_layer=FrozenBatchNorm2d if ini != "none" else nn.BatchNorm2d)
            if ini == "imagenet":
                r.load_state_dict(ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=False), strict=False)
            elif ini == "dino":
                r.load_state_dict(torch.hub.load_state_dict_from_url(DINO_R50_URL, map_location="cpu"), strict=False)
            return _resnet_fpn_extractor(r, tl if ini != "none" else 5), default_rpn(), ini
        except Exception as e:
            print(f"[init] '{ini}' failed ({type(e).__name__}: {str(e)[:120]}) -> fallback")
    raise RuntimeError("could not build backbone")


# ---------------------------- text prototypes (CLIP text encoder) ----------------------------
_TEXT_CACHE = WORK / "logs" / "clip_text.pt"   # on Drive: CLIP is loaded once, not every session
TEXT_INFO = dict(source=None, transformers=None, available=False, embedding_sha=None, error=None)


def _as_text_tensor(e):
    """CLIPModel.get_text_features returns a Tensor in most transformers versions and a ModelOutput in some.
    Never use `a or b` on tensors (ambiguous truth value)."""
    if torch.is_tensor(e):
        return e
    for name in ("text_embeds", "pooler_output"):
        v = getattr(e, name, None)
        if v is not None:
            return v
    return e[0]


def clip_text_embed(prompts):
    cache = torch.load(_TEXT_CACHE) if _TEXT_CACHE.exists() else {}
    todo = [p for p in prompts if p not in cache]
    if todo:
        src = C.CLIP_HF_NAME if INTERNET or os.path.isdir(C.CLIP_HF_NAME) else None
        if src is None:
            TEXT_INFO["error"] = "no internet and no local CLIP folder"
            return None
        try:
            import transformers
            from transformers import CLIPModel, CLIPTokenizer
            tok, m = CLIPTokenizer.from_pretrained(src), CLIPModel.from_pretrained(src).eval()
            with torch.no_grad():
                for i in range(0, len(todo), 64):
                    t = tok(todo[i:i + 64], padding=True, return_tensors="pt")
                    e = _as_text_tensor(m.get_text_features(**t))
                    for p, v in zip(todo[i:i + 64], F.normalize(e.float(), dim=-1)):
                        cache[p] = v
            torch.save(cache, _TEXT_CACHE)
            TEXT_INFO.update(source=src, transformers=transformers.__version__)
            del m
        except Exception as e:
            TEXT_INFO["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print("[text] CLIP text encoder unavailable:", TEXT_INFO["error"]); return None
    out = torch.stack([cache[p] for p in prompts])
    TEXT_INFO.update(available=True, embedding_sha=sha(out.numpy().round(5).tolist()), source=TEXT_INFO["source"] or C.CLIP_HF_NAME + " (cache)")
    return out


PROMPTS = ["an x-ray scan of a {}", "a {} inside an x-ray baggage image", "a photo of a {}"]


def text_prototypes(known):
    ps = [t.format(TEXT_NAME.get(c, c.replace("_", " "))) for c in known for t in PROMPTS]
    thr = ["an x-ray scan of a dangerous prohibited item", "a threat object hidden in luggage", "a prohibited item"]
    e = clip_text_embed(ps + thr)
    if e is None:
        return None
    cls = F.normalize(e[:len(ps)].view(len(known), len(PROMPTS), -1).mean(1), dim=-1)
    return dict(cls=cls, threat=F.normalize(e[len(ps):].mean(0), dim=-1))


# ---------------------------- modules ----------------------------
class SIRAAdapter(nn.Module):
    """Frequency-conditioned scanner invariance: F_L = LP(F), F_H = F - F_L,
    G = sigmoid(MLP([F_L, F_H])), F_inv = G*F_H + (1-G)*F_L;  out = F + gamma*proj(F_inv)."""
    def __init__(self, ch=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * ch, ch // 4, 1), nn.GELU(), nn.Conv2d(ch // 4, ch, 1))
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.last_gate = None

    def forward(self, x):
        L = F.avg_pool2d(x, 5, 1, 2, count_include_pad=False)
        H = x - L
        G = torch.sigmoid(self.gate(torch.cat([L, H], 1)))
        self.last_gate = G.detach().mean().item() if not self.training else None
        return x + self.gamma * self.proj(G * H + (1 - G) * L)


class OER(nn.Module):
    """Occlusion Evidence Router on the 7x7 RoI token grid: A(q,k_i) = q.k_i/sqrt(d) + lambda*log(r_i+eps)."""
    def __init__(self, ch=256, heads=4, n_tok=49):
        super().__init__()
        self.h = heads
        self.rel = nn.Sequential(nn.Linear(ch, 64), nn.GELU(), nn.Linear(64, 1))
        self.norm = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = nn.Linear(ch, ch)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, ch))
        self.lam = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))   # identity at initialisation: OER cannot hurt the pretrained head at step 0

    def forward(self, x):
        N, Cc, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                                   # N, HW, C
        r = torch.sigmoid(self.rel(t).float())                             # N, HW, 1  evidence reliability
        q, k, v = self.qkv(self.norm(t + self.pos)).chunk(3, -1)
        sh = lambda a: a.view(N, H * W, self.h, Cc // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(-1, -2)).float() / math.sqrt(Cc // self.h)
        att = att + self.lam * torch.log(r + 1e-4).transpose(1, 2).unsqueeze(1)   # bias per key token
        o = (att.softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(N, H * W, Cc)
        t = t + self.gamma * self.proj(o)
        return t.transpose(1, 2).reshape(N, Cc, H, W), r.squeeze(-1)


def pair_iou(a, b):
    lt, rb = torch.max(a[:, :2], b[:, :2]), torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    return inter / (area(a) + area(b) - inter).clamp(min=1e-6)


def bce_prob(p, t):
    """BCE on probabilities, autocast-safe."""
    p = p.float().clamp(1e-4, 1 - 1e-4)
    return -(t * p.log() + (1 - t) * (1 - p).log()).mean()


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


STPB_ALIASES = {"full": "vta", "text": "t", "attr": "a", "visual": "v"}


class STPB(nn.Module):
    """Prototype bank. mode = "none" (plain linear classifier) or any subset of "v" (visual), "t" (CLIP text),
    "a" (attribute) -> p_c = normalize(sum_k w_k p_c^k), w = softmax(learned). The superclass (hierarchy) loss is
    applied for every non-"none" mode so that the prototype-source ablation only varies the sources."""
    def __init__(self, known, D, mode, text=None):
        super().__init__()
        K, NA = len(known), len(ATTRS)
        mode = STPB_ALIASES.get(mode, mode)
        self.mode = mode
        A = torch.tensor(np.stack([attr_vector(c) for c in known]))
        self.register_buffer("A", A)
        sup = sorted({superclass(c) for c in known})
        M = torch.zeros(len(sup), K)
        for i, c in enumerate(known):
            M[sup.index(superclass(c)), i] = 1
        self.register_buffer("Msup", M / M.sum(1, keepdim=True).clamp(min=1))
        self.register_buffer("sup_of", torch.tensor([sup.index(superclass(c)) for c in known]))
        self.n_sup = len(sup)
        self.attr_head = nn.Linear(D, NA)
        self.threat = nn.Parameter(F.normalize(torch.randn(D), dim=0))
        self.register_buffer("ema", torch.zeros(K, D)); self.register_buffer("ema_n", torch.zeros(K))
        self.text_available = text is not None
        self.requested = set(mode) if mode != "none" else set()
        self.use_text = "t" in self.requested and text is not None
        self.use_attr = "a" in self.requested
        self.use_vis = "v" in self.requested or (self.requested and not (self.use_text or self.use_attr))
        if mode == "none":
            self.lin = nn.Linear(D, K)
            return
        self.P_V = nn.Parameter(F.normalize(torch.randn(K, D), dim=1))
        if self.use_text:
            self.register_buffer("T", text["cls"].float()); self.register_buffer("T_thr", text["threat"].float())
            self.text_proj = nn.Linear(self.T.shape[1], D)
        if self.use_attr:
            self.attr_emb = nn.Parameter(torch.randn(NA, D) * 0.1)
        self.mix = nn.Parameter(torch.zeros(3))
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def sources(self):
        return "".join(c for c, u in [("v", self.use_vis), ("t", self.use_text), ("a", self.use_attr)] if u) or "linear"

    def prototypes(self):
        if self.mode == "none":
            return F.normalize(self.lin.weight, dim=1)
        comps = []
        if self.use_vis:
            comps.append(F.normalize(self.P_V, dim=1))
        if self.use_text:
            comps.append(F.normalize(self.text_proj(self.T), dim=1))
        if self.use_attr:
            An = (self.A + 1e-3) / (self.A + 1e-3).sum(1, keepdim=True)
            comps.append(F.normalize(An @ self.attr_emb, dim=1))
        w = self.mix[:len(comps)].softmax(0)
        return F.normalize(sum(wi * ci for wi, ci in zip(w, comps)), dim=1)

    def mix_weights(self):
        names = [n for n, u in [("visual", self.use_vis), ("text", self.use_text), ("attribute", self.use_attr)] if u]
        return dict(zip(names, self.mix[:len(names)].softmax(0).tolist())) if self.mode != "none" else {}

    def threat_proto(self):
        t = F.normalize(self.threat, dim=0)
        if self.mode != "none" and self.use_text:
            t = t + F.normalize(self.text_proj(self.T_thr[None])[0], dim=0)
        return F.normalize(t, dim=0)

    def logits(self, z_raw, P=None):
        if self.mode == "none":
            return self.lin(z_raw)
        P = self.prototypes() if P is None else P
        return self.log_scale.exp().clamp(max=50) * F.normalize(z_raw, dim=-1) @ P.t().to(z_raw.dtype)

    def sup_logits(self, z_raw, P):
        return self.log_scale.exp().clamp(max=50) * F.normalize(z_raw.float(), dim=-1) @ F.normalize(self.Msup @ P.float(), dim=1).t()


def different_class_pairs(y, n):
    """n index pairs (i, j) with y[i] != y[j] GUARANTEED; None if fewer than 2 classes are present."""
    cls = y.unique()
    if len(cls) < 2:
        return None
    i = torch.randint(0, len(y), (n,), device=y.device)
    ci = torch.searchsorted(cls, y[i])
    cj = cls[(ci + torch.randint(1, len(cls), (n,), device=y.device)) % len(cls)]
    j = torch.empty_like(i)
    for c in cls:
        m = cj == c
        if m.any():
            members = (y == c).nonzero().squeeze(1)
            j[m] = members[torch.randint(0, len(members), (int(m.sum()),), device=y.device)]
    assert bool((y[i] != y[j]).all())
    return i, j


class CUS(nn.Module):
    """Counterfactual Unknown Synthesis: z -> (s: shape, m: material, g: function/geometry);
    each factor supervised by its own attribute group; z_U = Q([s_A, m_B, lam*g_A+(1-lam)*g_B])."""
    def __init__(self, D, d=64):
        super().__init__()
        self.d = d
        self.P, self.Q = nn.Linear(D, 3 * d), nn.Linear(3 * d, D)
        self.gidx = [[ATTRS.index(a) for a in g] for g in ATTR_GROUPS.values()]
        self.heads = nn.ModuleList([nn.Linear(d, len(g)) for g in self.gidx])

    def losses(self, z, A_rows):
        h = self.P(z); parts = h.split(self.d, 1)
        rec = (1 - F.cosine_similarity(self.Q(h), z.detach(), dim=-1)).mean()
        att = sum(F.binary_cross_entropy_with_logits(hd(p), A_rows[:, gi]) for hd, p, gi in zip(self.heads, parts, self.gidx))
        return rec, att / 3

    def synth(self, z, y, n):
        """Different-class factor mixing; returns None when the batch holds < 2 classes."""
        pr = different_class_pairs(y, n)
        if pr is None:
            return None
        i, j = pr
        s, m, g = self.P(z).split(self.d, 1)
        lam = torch.rand(len(i), 1, device=z.device)
        h = torch.cat([s[i], m[j], lam * g[i] + (1 - lam) * g[j]], 1)
        return self.Q(h)


class XGuard(nn.Module):
    def __init__(self, known, mcfg, init="coco", tl=3, img_size=640, D=256):
        super().__init__()
        self.known, self.K, self.m = list(known), len(known), dict(mcfg)
        self.backbone, self.rpn, self.init_used = build_backbone_rpn(init, tl)
        self.transform = GeneralizedRCNNTransform(img_size, img_size, IMAGENET_MEAN, IMAGENET_STD)
        self.sira = SIRAAdapter(256) if self.m["sira_freq"] else None
        self.roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2)
        self.oer = OER(256) if self.m["oer"] else None
        self.box_head = TwoMLPHead(256 * 49, 1024)
        self.reg, self.bg, self.embed = nn.Linear(1024, 4), nn.Linear(1024, 1), nn.Linear(1024, D)
        mode = STPB_ALIASES.get(self.m["stpb"], self.m["stpb"])
        text = text_prototypes(self.known) if "t" in mode and mode != "none" and not FORCE_NO_TEXT else None
        if "t" in mode and mode != "none" and text is None and C.REQUIRE_TEXT and not FORCE_NO_TEXT:
            raise RuntimeError(f"CLIP text prototypes required for STPB '{mode}' but unavailable: {TEXT_INFO['error']}")
        self.stpb = STPB(self.known, D, mode, text)
        self.diag = defaultdict(list)
        self.cus = CUS(D) if self.m["unk_gen"] == "cus" else None
        self.unk_head = mlp(D + 2, 1) if self.m["unk_gen"] != "none" else None
        self.unk_skipped = 0
        self.risk_head = mlp(D + 4, 1) if self.m["rcr"] else None
        self.box_coder = det_utils.BoxCoder((10.0, 10.0, 5.0, 5.0))
        self.matcher = det_utils.Matcher(0.5, 0.5, False)
        self.sampler = det_utils.BalancedPositiveNegativeSampler(512, 0.25)
        self.tau_U_formula = None

    # ---------------- shared pieces ----------------
    def features(self, xs, targets=None):
        il, tt = self.transform(list(xs), targets)
        f = self.backbone(il.tensors)
        if self.sira is not None:
            f = OrderedDict((k, self.sira(v)) for k, v in f.items())
        return il, tt, f

    def head(self, feats, boxes, sizes):
        x = self.roi_pool(feats, boxes, sizes)
        r = None
        if self.oer is not None:
            x, r = self.oer(x)
        h = self.box_head(x)
        z = self.embed(h)
        return dict(h=h, z=z, fg=self.stpb.logits(z), bgl=self.bg(h), reg=self.reg(h), r=r)

    def open_scores(self, out, P=None):
        probs = torch.cat([out["bgl"], out["fg"]], 1).float().softmax(1)
        o = 1 - probs[:, 0]
        pfg = probs[:, 1:]
        zn = F.normalize(out["z"].float(), dim=-1)
        P = self.stpb.prototypes().float() if P is None else P.float()
        s_k = (zn @ P.t()).max(1).values.add(1).div(2)
        s_t = (zn @ self.stpb.threat_proto().float()).add(1).div(2)
        if self.unk_head is not None:
            u = torch.sigmoid(self.unk_head(torch.cat([zn, s_k[:, None], s_t[:, None]], 1).to(out["z"].dtype)).float()).squeeze(1)
            U = o * u
        elif self.m["stpb"] != "none":
            U = o * s_t * (1 - s_k)                       # U_i = o_i * s_t * (1 - s_k)
        else:
            cond = pfg / pfg.sum(1, keepdim=True).clamp(min=1e-6)
            U = o * (1 - cond.max(1).values)              # objectness x MSP baseline
        ent = -(pfg.clamp(min=1e-8) * pfg.clamp(min=1e-8).log()).sum(1)
        return probs, o, s_k, s_t, U, ent

    def energy_entropy(self, out):
        """Standard uncertainty baselines: energy = -logsumexp(known logits); normalised entropy of p(known|fg)."""
        fg = out["fg"].float()
        E = -torch.logsumexp(fg, 1)
        q = fg.softmax(1)
        H = -(q * q.clamp(min=1e-8).log()).sum(1) / math.log(max(2, fg.shape[1]))
        return E, H

    def risk(self, z, conf, U, ent, area):
        x = torch.cat([F.normalize(z.float(), dim=-1), conf[:, None], U[:, None], ent[:, None], area[:, None]], 1)
        return self.risk_head(x.to(z.dtype)).float().squeeze(1)

    def sample(self, proposals, targets):
        P, L, RT, MG = [], [], [], []
        for p, t in zip(proposals, targets):
            gt, gl = t["boxes"], t["labels"]
            p = torch.cat([p, gt]) if len(gt) else p
            if len(gt) == 0:
                lab = torch.zeros(len(p), dtype=torch.int64, device=p.device); mg = torch.zeros_like(p)
            else:
                idx = self.matcher(box_iou(gt, p)); cl = idx.clamp(min=0)
                lab = gl[cl] + 1
                lab[idx == self.matcher.BELOW_LOW_THRESHOLD] = 0
                lab[idx == self.matcher.BETWEEN_THRESHOLDS] = -1
                mg = gt[cl]
            pos, neg = self.sampler([lab])
            keep = torch.where(pos[0] | neg[0])[0]
            P.append(p[keep]); L.append(lab[keep]); MG.append(mg[keep])
        RT = self.box_coder.encode(MG, P)
        return P, torch.cat(L), torch.cat(RT), torch.cat(MG)

    # ---------------- training ----------------
    def forward_train(self, xs, targets, xs2=None, vis2=None):
        W = C.LOSS_W
        B = len(xs)
        two = xs2 is not None
        tg = [dict(boxes=t["boxes"], labels=t["labels"]) for t in targets]
        il, tt, feats = self.features(list(xs) + (list(xs2) if two else []), tg + ([dict(t) for t in tg] if two else []))
        # detection loss on view 1, and on view 2 only if view 2 is NOT occluded (occluded boxes are noisy labels)
        det_on_v2 = two and C.VIEW2_DET and not self.m.get("occ")
        nd = 2 * B if det_on_v2 else B
        if nd < len(tt):
            il_d = ImageList(il.tensors[:nd], il.image_sizes[:nd])
            feats_d = OrderedDict((k, v[:nd]) for k, v in feats.items())
            tt_d = tt[:nd]
        else:
            il_d, feats_d, tt_d = il, feats, tt
        proposals, rpn_l = self.rpn(il_d, feats_d, tt_d)
        props, lab, rt, mg = self.sample(proposals, tt_d)
        out = self.head(feats_d, props, il_d.image_sizes)
        logits = torch.cat([out["bgl"], out["fg"]], 1).float()
        L = dict(rpn_obj=rpn_l["loss_objectness"], rpn_box=rpn_l["loss_rpn_box_reg"])
        L["cls"] = F.cross_entropy(logits, lab)
        pos = lab > 0
        L["box"] = F.smooth_l1_loss(out["reg"][pos].float(), rt[pos].float(), beta=1 / 9, reduction="sum") / max(1, lab.numel())
        z, y = out["z"].float(), lab - 1
        zp, yp = z[pos], y[pos]
        st = self.stpb
        P = st.prototypes().float()
        # ---------- STPB ----------
        if self.m["stpb"] != "none" and pos.any():
            L["proto"] = F.binary_cross_entropy_with_logits(st.attr_head(zp), st.A[yp])
            thr = st.threat_proto().float()
            zb = F.normalize(z[lab == 0][:256], dim=-1)
            L["threat"] = (1 - F.normalize(zp, dim=-1) @ thr).mean() + (F.relu(zb @ thr)).mean()
            if st.n_sup > 1:
                L["proto"] = L["proto"] + F.cross_entropy(st.sup_logits(zp, P).float(), st.sup_of[yp])
            with torch.no_grad():
                for c in yp.unique():
                    zc = F.normalize(zp[yp == c], dim=-1).mean(0)
                    st.ema[c] = F.normalize(0.99 * st.ema[c] + 0.01 * zc if st.ema_n[c] > 0 else zc, dim=0)
                    st.ema_n[c] += 1
            if st.use_vis:
                seen = st.ema_n > 0
                L["proto"] = L["proto"] + (1 - (F.normalize(st.P_V[seen], dim=1) * st.ema[seen]).sum(1)).mean()
        # ---------- unknown discovery: cus | gaussian | mixup | slerp ----------
        gen = {"random": "gaussian"}.get(self.m["unk_gen"], self.m["unk_gen"])
        if self.unk_head is not None and pos.sum() >= 2:
            n1 = int((torch.cat([torch.full((len(p),), i) for i, p in enumerate(props)]) < B)[pos.cpu()].sum())
            zr, yr = zp[:max(2, n1)], yp[:max(2, n1)]
            n = min(256, 2 * len(zr))
            zu = None
            if gen == "cus":
                rec, att = self.cus.losses(zr, st.A[yr])
                L["rec"] = rec; L["proto"] = L.get("proto", 0) + att
                zu = self.cus.synth(zr, yr, n)
            elif gen == "gaussian":   # Gaussian feature noise around known embeddings
                zu = F.normalize(zr, dim=-1)[torch.randint(0, len(zr), (n,), device=zr.device)] + 0.5 * torch.randn(n, zr.shape[1], device=zr.device) / math.sqrt(zr.shape[1])   # noise norm ~0.5
            elif gen in ("mixup", "slerp"):
                pr = different_class_pairs(yr, n)
                if pr is not None:
                    a_, b_ = F.normalize(zr[pr[0]], dim=-1), F.normalize(zr[pr[1]], dim=-1)
                    lam = torch.rand(len(pr[0]), 1, device=zr.device) * 0.4 + 0.3
                    if gen == "mixup":
                        zu = lam * a_ + (1 - lam) * b_
                    else:             # manifold-aware: spherical interpolation stays on the unit hypersphere
                        om = torch.acos((a_ * b_).sum(1, keepdim=True).clamp(-0.9999, 0.9999))
                        zu = (torch.sin((1 - lam) * om) * a_ + torch.sin(lam * om) * b_) / torch.sin(om).clamp(min=1e-4)
            if zu is None or len(zu) == 0:
                self.unk_skipped += 1
            else:
                zun, zrn = F.normalize(zu.float(), dim=-1), F.normalize(zr.float(), dim=-1)
                thr = st.threat_proto().float()
                sk_r, st_r = (zrn @ P.t()).max(1).values.float(), (zrn @ thr).float()
                sk_u, st_u = (zun @ P.t()).max(1).values.float(), (zun @ thr).float()
                if gen in ("cus", "slerp"):   # keep threat-like counterfactuals unlike every known prototype
                    keep = (sk_u < torch.quantile(sk_r.detach().float(), 0.25)) & (st_u > torch.quantile(st_r.detach().float(), 0.10))
                    if keep.sum() < 4:
                        keep = sk_u <= sk_u.median()
                    zun, sk_u, st_u = zun[keep], sk_u[keep], st_u[keep]
                if len(zun) == 0:
                    self.unk_skipped += 1
                else:
                    self._last_synth = zun.detach()
                    self.diag["synth_max_known_cos"].append(float(sk_u.mean()))
                    self.diag["real_max_known_cos"].append(float(sk_r.mean()))
                    f_u = torch.cat([zun, ((sk_u + 1) / 2)[:, None], ((st_u + 1) / 2)[:, None]], 1)
                    f_r = torch.cat([zrn, ((sk_r + 1) / 2)[:, None], ((st_r + 1) / 2)[:, None]], 1)
                    lu, lr = self.unk_head(f_u.to(out["z"].dtype)).float(), self.unk_head(f_r.to(out["z"].dtype)).float()
                    L["unknown"] = F.binary_cross_entropy_with_logits(lu, torch.ones_like(lu)) + \
                        F.binary_cross_entropy_with_logits(lr, torch.zeros_like(lr))
                    lk = st.logits(zun.to(out["z"].dtype)).float()
                    L["ent"] = -(lk.log_softmax(1)).mean()   # push synthetic unknowns away from every known class
        # ---------- invariance / occlusion consistency on GT RoIs ----------
        if two:
            gtb = [t["boxes"] for t in tt]
            has = sum(len(b) for b in gtb[:B]) > 0
            if self.m["sira_cf"]:
                w_lvl = []
                for k in ["0", "1", "2", "3"]:
                    f1, f2 = feats[k][:B].detach().float(), feats[k][B:].float()
                    cs = F.cosine_similarity(f1, f2, dim=1)[:, None]
                    if vis2 is not None:
                        v = F.interpolate(self._pad_vis(vis2, il), size=cs.shape[-2:], mode="bilinear", align_corners=False)
                        w_lvl.append(((1 - cs) * v).sum() / v.sum().clamp(min=1))
                    else:
                        w_lvl.append((1 - cs).mean())
                L["sira"] = sum(w_lvl) / len(w_lvl)
            if has:
                o_gt = self.head(feats, gtb, il.image_sizes)
                n1 = sum(len(b) for b in gtb[:B])
                z1, z2 = o_gt["z"][:n1].float(), o_gt["z"][n1:].float()
                L["cons_obj"] = (1 - F.cosine_similarity(z1.detach(), z2, dim=-1)).mean()
                p1 = torch.cat([o_gt["bgl"], o_gt["fg"]], 1)[:n1].float().softmax(1).detach()
                lp2 = torch.cat([o_gt["bgl"], o_gt["fg"]], 1)[n1:].float().log_softmax(1)
                L["cons"] = F.kl_div(lp2, p1, reduction="batchmean")
                if self.oer is not None and vis2 is not None and o_gt["r"] is not None:
                    vp = self._pad_vis(vis2, il)
                    vr = roi_align(vp, gtb[B:], output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True).flatten(1)
                    r1, r2 = o_gt["r"][:n1], o_gt["r"][n1:]
                    L["rel"] = bce_prob(r2, vr.clamp(0, 1)) + bce_prob(r1, torch.ones_like(r1).float())
                    with torch.no_grad():
                        occl, visi = vr < 0.5, vr >= 0.5
                        if occl.any():
                            self.diag["oer_r_occluded"].append(float(r2[occl].float().mean()))
                        if visi.any():
                            self.diag["oer_r_visible"].append(float(r2[visi].float().mean()))
        # ---------- RCR: learn P(detection incorrect) ----------
        if self.risk_head is not None:
            with torch.no_grad():
                probs, o, s_k, s_t, U, ent = self.open_scores({k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()}, P.detach())
                conf, pc = probs[:, 1:].max(1)
                db = self.box_coder.decode(out["reg"].detach().float(), props).reshape(-1, 4)
                iou = pair_iou(db, mg)
                correct = (lab > 0) & (pc == lab - 1) & (iou >= 0.5)
                pb = torch.cat(props)
                area = ((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])).clamp(min=1).log() / 14
            rl = self.risk(out["z"].detach(), conf, U, ent, area)
            tgt = (~correct).float()
            pw = (tgt.numel() - tgt.sum()).clamp(min=1) / tgt.sum().clamp(min=1)
            L["risk"] = F.binary_cross_entropy_with_logits(rl, tgt, weight=torch.where(tgt > 0, 1.0, pw.clamp(max=20)).detach())
        return L

    def _pad_vis(self, vis, il):
        Hp, Wp = il.tensors.shape[-2:]
        out = torch.zeros(len(vis), 1, Hp, Wp, device=il.tensors.device)
        for i, (v, (h, w)) in enumerate(zip(vis, il.image_sizes[len(il.image_sizes) - len(vis):])):
            if v.shape[-2:] != (h, w):
                v = F.interpolate(v[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            out[i, :, :h, :w] = v
        return out

    # ---------------- inference ----------------
    @torch.no_grad()
    def forward(self, xs):
        sizes0 = [x.shape[-2:] for x in xs]
        il, _, feats = self.features(xs)
        props, _ = self.rpn(il, feats)
        out = self.head(feats, props, il.image_sizes)
        probs, o, s_k, s_t, U, ent = self.open_scores(out)
        boxes = self.box_coder.decode(out["reg"].float(), props).reshape(-1, 4)
        pfg = probs[:, 1:]
        conf = pfg.max(1).values
        pb = torch.cat(props)
        area = ((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])).clamp(min=1).log() / 14
        R = torch.sigmoid(self.risk(out["z"], conf, U, ent, area)) if self.risk_head is not None else 1 - conf
        E, Hn = self.energy_entropy(out)
        res, start = [], 0
        for i, p in enumerate(props):
            n = len(p); sl = slice(start, start + n); start += n
            (h, w), (h0, w0) = il.image_sizes[i], sizes0[i]
            b = boxes[sl].clone()
            b[:, 0::2] = b[:, 0::2].clamp(0, w); b[:, 1::2] = b[:, 1::2].clamp(0, h)
            b = b * torch.tensor([w0 / w, h0 / h, w0 / w, h0 / h], device=b.device)
            ok = ((b[:, 2] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 1]) > 1)
            sc = pfg[sl]
            roi, cls = torch.where((sc > C.SCORE_THR) & ok[:, None])
            s = sc[roi, cls]
            k = batched_nms(b[roi], s, cls, 0.5)[: C.DETS_PER_IMG]
            roi, cls, s = roi[k], cls[k], s[k]
            u = U[sl]
            ru = torch.where((u > C.SCORE_THR) & ok)[0]
            ku = nms(b[ru], u[ru], 0.5)[: C.UNK_DETS_PER_IMG]
            ru = ru[ku]
            res.append(dict(boxes=b[roi].cpu(), scores=s.cpu(), labels=cls.cpu(), U=u[roi].cpu(), R=R[sl][roi].cpu(),
                            E=E[sl][roi].cpu(), H=Hn[sl][roi].cpu(),
                            ub=b[ru].cpu(), us=u[ru].cpu(), uR=R[sl][ru].cpu(), uE=E[sl][ru].cpu(), uH=Hn[sl][ru].cpu()))
        return res

    @torch.no_grad()
    def embed_boxes(self, xs, boxes):
        il, _, feats = self.features(xs)
        sc = [torch.tensor([w1 / w0, h1 / h0, w1 / w0, h1 / h0], device=b.device) for b, x, (h1, w1) in
              zip(boxes, xs, il.image_sizes) for (h0, w0) in [x.shape[-2:]]]
        out = self.head(feats, [b * s for b, s in zip(boxes, sc)], il.image_sizes)
        return out["z"].float()


class TVDetector(nn.Module):
    """Closed-set torchvision baselines (FCOS / RetinaNet / Faster R-CNN-v1) under the same pipeline.
    Unknown score = 1 - max score (MSP-style), risk = 1 - score."""
    def __init__(self, kind, known, init, tl, img_size):
        super().__init__()
        from torchvision.models.detection import fcos_resnet50_fpn, retinanet_resnet50_fpn
        from torchvision.models import ResNet50_Weights
        self.known, self.K, self.m = list(known), len(known), dict(kind=kind)
        wb = ResNet50_Weights.IMAGENET1K_V1 if (init != "none" and INTERNET) else None
        builder = dict(fcos=fcos_resnet50_fpn, retinanet=retinanet_resnet50_fpn)[kind]
        try:
            self.net = builder(weights=None, weights_backbone=wb, num_classes=self.K + 1,
                               trainable_backbone_layers=tl if wb else 5, min_size=img_size, max_size=img_size)
            self.init_used = "imagenet" if wb else "none"
        except Exception as e:
            print("[init] backbone weights failed:", e)
            self.net = builder(weights=None, weights_backbone=None, num_classes=self.K + 1, min_size=img_size, max_size=img_size)
            self.init_used = "none"
        self.net.score_thresh = C.SCORE_THR
        self.net.detections_per_img = C.DETS_PER_IMG

    def forward_train(self, xs, targets, **kw):
        L = self.net(list(xs), [dict(boxes=t["boxes"], labels=t["labels"] + 1) for t in targets])
        return {k: v for k, v in L.items()}

    @torch.no_grad()
    def forward(self, xs):
        res = []
        for d in self.net(list(xs)):
            keep = d["labels"] > 0
            b, s, l = d["boxes"][keep].cpu(), d["scores"][keep].cpu(), d["labels"][keep].cpu() - 1
            nan = torch.full_like(s, float("nan"))
            res.append(dict(boxes=b, scores=s, labels=l, U=1 - s, R=1 - s, E=nan, H=nan, ub=b, us=1 - s, uR=1 - s, uE=nan, uH=nan))
        return res


FORCE_NO_TEXT = False   # set while re-loading a model that was trained without text prototypes

_B = dict(sira_freq=0, sira_cf=0, oer=0, occ=0, stpb="none", unk_gen="none", rcr=0, aug="none")
_S, _O, _P, _U, _R = dict(sira_freq=1, sira_cf=1), dict(oer=1, occ=1), dict(stpb="vta"), dict(unk_gen="cus"), dict(rcr=1)


def _v(*parts):
    d = dict(_B)
    for p_ in parts:
        d.update(p_)
    return d


MODEL_VARIANTS = {   # component ablation matrix (review: independent + cumulative + leave-one-out)
    "base": _v(), "sira": _v(_S), "oer": _v(_O), "stpb": _v(_P),
    "sira+oer": _v(_S, _O), "sira+oer+stpb": _v(_S, _O, _P), "sira+oer+stpb+cus": _v(_S, _O, _P, _U),
    "xguard": _v(_S, _O, _P, _U, _R),
    "xguard-sira": _v(_O, _P, _U, _R), "xguard-oer": _v(_S, _P, _U, _R),
    "xguard-cus": _v(_S, _O, _P, _R), "xguard-rcr": _v(_S, _O, _P, _U),
}


def variant(base="xguard", **kw):
    d = dict(MODEL_VARIANTS[base]); d.update(kw); return d


def build_model(spec, known):
    kind = spec.get("kind", "xguard")
    if kind in ("fcos", "retinanet"):
        return TVDetector(kind, known, spec.get("init", C.INIT), spec.get("tl", C.TRAINABLE_LAYERS), C.IMG)
    return XGuard(known, spec["mcfg"], spec.get("init", C.INIT), spec.get("tl", C.TRAINABLE_LAYERS), C.IMG, C.EMB_DIM)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
