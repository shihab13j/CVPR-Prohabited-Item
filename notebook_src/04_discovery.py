# ==============================================================================
# 3. DATASET DISCOVERY (raw inputs only; skipped when a pack exists) + FORMAT-AGNOSTIC PARSERS
#    Supported: COCO-JSON (PIDray, CLCXray), Pascal-VOC XML (SIXray), per-line TXT
#    "img cls x1 y1 x2 y2" (OPIXray, HiXray), YOLO txt (+data.yaml/classes.txt), CSV.
# ==============================================================================
import xml.etree.ElementTree as ET
import tarfile
from concurrent.futures import ThreadPoolExecutor

DS_KEYS = ["sixray", "opixray", "hixray", "pidray", "clcxray"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANN_EXT = {".xml", ".json", ".txt", ".csv", ".yaml", ".yml", ".names"}
NEG_DATASETS = {"sixray"}   # images without boxes in these datasets are used as clean (negative) bags
ARCHIVE_EXT = (".zip", ".tar", ".gz", ".tgz", ".rar", ".7z")

# Class-name fingerprints: identify a Kaggle mirror by its CONTENT when its folder name does not say which
# dataset it is (e.g. "xray-contraband-detection" = PIDray). >= 2 hits of the signature classes are required.
FINGERPRINT = OrderedDict(
    pidray={"handcuffs", "baton", "sprayer", "bullet", "powerbank"},
    hixray={"portable_charger_1", "portable_charger_2", "mobile_phone", "laptop", "tablet", "cosmetic", "water", "nonmetallic_lighter"},
    opixray={"folding_knife", "straight_knife", "utility_knife", "multitool_knife"},
    clcxray={"swiss_army_knife", "cans", "carton_drinks", "glass_bottle", "plastic_bottle", "vacuum_cup", "spray_cans", "tin", "blade", "dagger"},
    sixray={"gun", "knife", "wrench", "pliers", "scissors", "hammer"},
)
# Used ONLY for YOLO mirrors that ship no data.yaml / classes.txt (official class order; verify in the printout!)
YOLO_DEFAULT_NAMES = {
    "sixray": ["Gun", "Knife", "Wrench", "Pliers", "Scissors", "Hammer"],
    "opixray": ["Folding_Knife", "Straight_Knife", "Scissor", "Utility_Knife", "Multi-tool_Knife"],
    "hixray": ["Portable_Charger_1", "Portable_Charger_2", "Mobile_Phone", "Laptop", "Tablet", "Cosmetic", "Water", "Nonmetallic_Lighter"],
    "pidray": ["Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench", "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter"],
}


# ---------- resume: bring finished runs from a previous notebook version ----------
def resume_from_input():
    if PLATFORM != "kaggle" or not RESUME_FROM_INPUT or not os.path.isdir(INPUT_ROOT):
        return
    n = 0
    for cand in glob.glob(os.path.join(INPUT_ROOT, "**", "xguard", "runs"), recursive=True)[:5]:
        if str(WORK) in cand:
            continue
        for rd in Path(cand).iterdir():
            dst = WORK / "runs" / rd.name
            if rd.is_dir() and not dst.exists():
                shutil.copytree(rd, dst); n += 1
        for extra in ["audit", "cache", "extracted"]:
            src = Path(cand).parent / extra
            if src.exists():
                shutil.copytree(src, WORK / extra, dirs_exist_ok=True)
    if n:
        print(f"[resume] copied {n} run folders from previous output")




# ---------- archive extraction (optional; Kaggle normally auto-extracts .zip uploads) ----------
EXTRACT_DIR = Path(os.environ.get("XGUARD_EXTRACT_DIR", "/tmp/xguard_extract"))   # /tmp is not limited to 20 GB like /kaggle/working


def list_archives(root, limit=20, max_depth=2):
    """Archives in the top levels of an input (cheap: does not walk million-file trees)."""
    out = []
    for dirpath, dirnames, files in os.walk(root):
        if len(Path(dirpath).relative_to(root).parts) >= max_depth:
            dirnames[:] = []
        out += [os.path.join(dirpath, f) for f in files if f.lower().endswith(ARCHIVE_EXT)]
        if len(out) >= limit:
            break
    return out


def maybe_extract(root):
    arcs = list_archives(root)
    if not arcs:
        return root
    if not EXTRACT_ARCHIVES:
        print(f"   ⚠️  {root} contains archive(s) {[os.path.basename(a) for a in arcs[:3]]} that Kaggle did not extract. "
              "Use an already-extracted mirror, or set EXTRACT_ARCHIVES=True (extracts to /tmp).")
        return root
    out = EXTRACT_DIR / Path(root).name
    if not (out / ".done").exists():
        out.mkdir(parents=True, exist_ok=True)
        for a in arcs:
            print("extracting", a)
            if a.lower().endswith((".rar", ".7z")):
                subprocess.run(["7z", "x", "-y", f"-o{out}", a], check=False)
            else:
                shutil.unpack_archive(a, str(out))
        (out / ".done").write_text("ok")
    return str(out)


# ---------- multi-part tar streams (e.g. SIXRAY_P1/P2/P3 = JPEGImage.tar.gz00 ... gz15) ----------
MULTIPART = re.compile(r"^(?P<base>.+?\.(?:tar\.gz|tgz|tar|tar\.bz2|tar\.xz))[._-]?(?:part)?[._-]?(?P<num>\d{1,3})$", re.I)


def multipart_sets(roots):
    parts = defaultdict(list)
    for r in roots:
        for dirpath, dirnames, files in os.walk(r):
            if len(Path(dirpath).relative_to(r).parts) >= 2:
                dirnames[:] = []
            for f in files:
                m = MULTIPART.match(f)
                if m:
                    parts[m.group("base")].append((int(m.group("num")), os.path.join(dirpath, f)))
    return {b: [p for _, p in sorted(v)] for b, v in parts.items()}


class ConcatReader(io.RawIOBase):
    """Reads several files as one continuous byte stream (split archives), without copying them."""
    def __init__(self, paths):
        self.paths, self.i, self.f = paths, 0, open(paths[0], "rb")

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close(); self.i += 1
            if self.i >= len(self.paths):
                return 0
            self.f = open(self.paths[self.i], "rb")


def stream_extract(paths, out_dir, want_stems, n_neg, neg_prefix="n"):
    """Stream through a (split) tar archive once and write only the wanted files:
    images referenced by annotations (want_stems) + the first n_neg clean bags (name starts with neg_prefix)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(want_stems); got_pos = got_neg = seen = 0; t0 = time.time()
    try:
      with tarfile.open(fileobj=io.BufferedReader(ConcatReader(paths), 1 << 22), mode="r|*") as tf:
        for m in tf:
            seen += 1
            if seen % 100000 == 0:
                print(f"   … {seen} members scanned, {got_pos} annotated + {got_neg} clean extracted ({(time.time() - t0) / 60:.1f} min)")
            if not m.isfile():
                continue
            b = os.path.basename(m.name); st, ext = os.path.splitext(b.lower())
            if ext not in IMG_EXT:
                continue
            is_pos = st in want
            is_neg = (not is_pos) and got_neg < n_neg and st.startswith(neg_prefix)
            if is_pos or is_neg:
                with open(out_dir / b, "wb") as f:
                    f.write(tf.extractfile(m).read())
                if is_pos:
                    want.discard(st); got_pos += 1
                else:
                    got_neg += 1
            if not want and got_neg >= n_neg:
                break
    except (tarfile.ReadError, EOFError, OSError) as e:   # only some parts downloaded -> stream ends early
        print(f"   archive stream ended early ({type(e).__name__}): partial download, keeping what was extracted")
    print(f"   extracted {got_pos} annotated + {got_neg} clean images in {(time.time() - t0) / 60:.1f} min; "
          f"{len(want)} annotated images not found in the archive")


def prepare_multipart(key, roots):
    """If a dataset is shipped as split tar parts, extract what is needed into /kaggle/working (persists across sessions)."""
    sets = multipart_sets(roots)
    if not sets or SIXRAY_TAR_EXTRACT is False:
        return roots
    out = CACHE / "extracted" / f"{key}_from_tar"
    imgs, anns = index_files(roots)
    have = {os.path.splitext(os.path.basename(p))[0].lower() for p in imgs}
    ann_stems = {os.path.splitext(os.path.basename(a))[0].lower() for a in anns if a.lower().endswith((".xml", ".txt"))}
    want = {s for s in ann_stems - have if key != "sixray" or re.match(r"^p\d+", s)}
    n_neg = (C.N_NEG_TRAIN + C.N_NEG_EVAL) if key in NEG_DATASETS else 0
    marker = out / f".done_{stable_hash(','.join(sorted(want)) + str(n_neg))}"   # new annotations -> extract again
    if marker.exists():
        return roots + [str(out)]
    if out.exists():
        want -= {os.path.splitext(f)[0].lower() for f in os.listdir(out)}
        cache = CACHE / f"index_{stable_hash(str(out))}.pkl"
        if cache.exists():
            cache.unlink()
    print(f"[{key}] split archive {[os.path.basename(p) for p in list(sets.values())[0][:3]]}… ({sum(len(v) for v in sets.values())} parts): "
          f"streaming {len(want)} annotated images + {n_neg} clean bags into {out}")
    if not want and key == "sixray":
        print("   ⚠️  no SIXray box annotations attached -> only clean bags are extracted. Attach an annotated SIXray "
              "(e.g. khanhbtq99/sixray, or SIXray-D / official positive XMLs uploaded as a dataset named '*sixray*').")
    n_have_neg = sum(1 for f in os.listdir(out) if f.lower().startswith("n")) if out.exists() else 0
    for base, paths in sets.items():
        stream_extract(paths, out, want, max(0, n_neg - n_have_neg))
    marker.write_text("ok")
    return roots + [str(out)]


# ---------- locate dataset roots (one Kaggle dataset = one root; a dataset may span several roots) ----------
def kaggle_dataset_dirs():
    """/kaggle/input/<slug>  and the newer layout /kaggle/input/datasets/<owner>/<slug>."""
    out = []
    if not os.path.isdir(INPUT_ROOT):
        return out
    for d in sorted(os.scandir(INPUT_ROOT), key=lambda e: e.name):
        if not d.is_dir():
            continue
        if d.name in ("datasets", "competitions"):
            for o in sorted(os.scandir(d.path), key=lambda e: e.name):
                if o.is_dir():
                    out += [x.path for x in sorted(os.scandir(o.path), key=lambda e: e.name) if x.is_dir()] if d.name == "datasets" else [o.path]
        elif d.name not in ("models", "notebooks"):
            out.append(d.path)
    # skip previous XGuard outputs (resume inputs) and weight-only inputs
    return [r for r in out if not glob.glob(os.path.join(r, "**", "xguard", "runs"), recursive=True)[:1]]


def name_key(path):
    n = re.sub(r"[^a-z0-9]", "", Path(path).name.lower())
    hits = [k for k in DS_KEYS if k in n]
    return hits[0] if len(hits) == 1 else None


def find_dataset_roots():
    groups, unnamed = defaultdict(list), []
    for k, v in DATASET_ROOTS.items():
        if v:
            groups[k] = [v] if isinstance(v, str) else list(v)
    manual = {os.path.normpath(p) for v in groups.values() for p in v}
    for r in kaggle_dataset_dirs():
        if os.path.normpath(r) in manual:
            continue
        k = name_key(r)
        if k and not DATASET_ROOTS.get(k):
            groups[k].append(r)
        elif not k:
            unnamed.append(r)
    groups = {k: [maybe_extract(r) for r in v] for k, v in groups.items()}
    return groups, [maybe_extract(r) for r in unnamed]


def print_tree(root, max_depth=3, max_children=8):
    root = Path(root)
    print(f"📁 {root}")

    def rec(p, depth, prefix):
        try:
            entries = sorted(os.scandir(p), key=lambda e: (not e.is_dir(), e.name))
        except Exception:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        ext = Counter(Path(f.name).suffix.lower() for f in files)
        if files:
            ex = ", ".join(f"{v}×{k or '(none)'}" for k, v in ext.most_common(5))
            print(f"{prefix}  {len(files)} files [{ex}]  e.g. {files[0].name}")
        if depth >= max_depth:
            if dirs:
                print(f"{prefix}  … {len(dirs)} sub-folders")
            return
        for d in dirs[:max_children]:
            print(f"{prefix}├─ {d.name}/")
            rec(d.path, depth + 1, prefix + "│  ")
        if len(dirs) > max_children:
            print(f"{prefix}└─ … +{len(dirs) - max_children} more folders")
    rec(root, 0, "")


def index_files(roots):
    if isinstance(roots, (list, tuple)):
        imgs, anns = [], []
        for r in roots:
            a, b = index_files(r); imgs += a; anns += b
        return imgs, anns
    root = roots
    cache = CACHE / f"index_{stable_hash(root)}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    imgs, anns = [], []
    for dirpath, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT:
                imgs.append(os.path.join(dirpath, f))
            elif ext in ANN_EXT:
                anns.append(os.path.join(dirpath, f))
    out = (imgs, anns)
    if len(imgs) > 20000:
        print(f"   indexed {len(imgs)} images under {root}")
    pickle.dump(out, open(cache, "wb"))
    return out


# ---------- split / subset inference from a path ----------
SPLIT_WORDS = [("train", "train"), ("trainval", "train"), ("valid", "val"), ("validation", "val"), ("val", "val"), ("test", "test"),
               ("easy", "test"), ("hard", "test"), ("hidden", "test")]


def _rel(path, root):
    roots = root if isinstance(root, (list, tuple)) else [root]
    for r in roots:
        if os.path.normpath(path).startswith(os.path.normpath(r) + os.sep):
            return os.path.relpath(path, r)
    return os.path.basename(path)


def infer_split(path, root):
    rel = _rel(path, root).lower()
    toks = re.split(r"[\\/_\-\.\s]+", rel)
    split, subsets = None, []
    for t in toks:
        for w, s in SPLIT_WORDS:
            if t == w or (t.startswith(w) and w in ("train", "test") and len(t) <= len(w) + 6):
                split = s
        if t in ("easy", "hard", "hidden"):
            subsets.append(t)
    return split, subsets


class ImageIndex:
    def __init__(self, paths, root):
        self.root = root
        self.by_name, self.by_stem = defaultdict(list), defaultdict(list)
        self.by_abs = {}
        for p in paths:
            b = os.path.basename(p)
            self.by_name[b.lower()].append(p)
            self.by_stem[os.path.splitext(b)[0].lower()].append(p)
            self.by_abs[os.path.normpath(p).lower()] = p

    def resolve(self, name, hint=None):
        if name is None:
            return None
        name = str(name).strip().replace("\\", "/")
        if os.path.isabs(name) and os.path.normpath(name).lower() in self.by_abs:
            return self.by_abs[os.path.normpath(name).lower()]
        roots = self.root if isinstance(self.root, (list, tuple)) else [self.root]
        bases = ([os.path.dirname(hint), os.path.dirname(os.path.dirname(hint))] if hint else []) + list(roots)
        for base in bases:
            q = os.path.normpath(os.path.join(base, name)).lower()
            if q in self.by_abs:
                return self.by_abs[q]
        b = os.path.basename(name).lower()
        c = self.by_name.get(b) or self.by_stem.get(os.path.splitext(b)[0]) or self.by_stem.get(b)
        if not c:
            return None
        if len(c) == 1 or hint is None:
            return c[0]
        hs, _ = infer_split(hint, self.root)
        hparts = set(re.split(r"[\\/]", _rel(hint, self.root).lower()))

        def score(p):
            ps, _ = infer_split(p, self.root)
            parts = set(re.split(r"[\\/]", _rel(p, self.root).lower()))
            return (ps == hs) * 10 + len(parts & hparts)
        return max(c, key=score)


# ---------- annotation parsers: each returns list of dict(img, path, split, subsets, objs=[(cls,x1,y1,x2,y2)], w, h) ----------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_coco(files, idx, root):
    out = []
    for jf in files:
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d):
            continue
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        split, subsets = infer_split(jf, root)
        by_img = defaultdict(list)
        for a in d["annotations"]:
            if a.get("iscrowd", 0):
                continue
            x, y, w, h = a["bbox"]
            by_img[a["image_id"]].append((cats.get(a["category_id"], str(a["category_id"])), x, y, x + w, y + h))
        for im in d["images"]:
            p = idx.resolve(im["file_name"], jf)
            if p is None:
                continue
            ps, psub = infer_split(p, root)
            out.append(dict(img=os.path.splitext(os.path.basename(p))[0], path=p, split=split or ps,
                            subsets=sorted(set(subsets + psub)), objs=by_img.get(im["id"], []),
                            w=im.get("width"), h=im.get("height"), src=os.path.basename(jf)))
    return out


def parse_voc(files, idx, root):
    out = []
    for xf in files:
        try:
            r = ET.parse(xf).getroot()
        except Exception:
            continue
        objs = []
        for o in r.iter("object"):
            n = o.findtext("name"); bb = o.find("bndbox")
            if n is None or bb is None:
                continue
            v = [_safe_float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if None in v:
                continue
            objs.append((n.strip(), *v))
        fn = r.findtext("filename") or os.path.splitext(os.path.basename(xf))[0]
        p = idx.resolve(fn, xf) or idx.resolve(os.path.splitext(os.path.basename(xf))[0], xf)
        if p is None:
            continue
        sz = r.find("size")
        w = _safe_float(sz.findtext("width")) if sz is not None else None
        h = _safe_float(sz.findtext("height")) if sz is not None else None
        s1, sub1 = infer_split(xf, root); s2, sub2 = infer_split(p, root)
        out.append(dict(img=os.path.splitext(os.path.basename(p))[0], path=p, split=s1 or s2,
                        subsets=sorted(set(sub1 + sub2)), objs=objs, w=w or None, h=h or None, src="voc"))
    return out


TXT_LINE5 = re.compile(r"^\s*([A-Za-z][\w\-]*)[\s,;]+(-?[\d.]+)[\s,;]+(-?[\d.]+)[\s,;]+(-?[\d.]+)[\s,;]+(-?[\d.]+)\s*$")
TXT_LINE = re.compile(r"^\s*(\S+?)[\s,;]+([A-Za-z][\w\-\s]*?)[\s,;]+(-?[\d.]+)[\s,;]+(-?[\d.]+)[\s,;]+(-?[\d.]+)[\s,;]+(-?[\d.]+)\s*$")


def parse_txt(files, idx, root):
    per = {}
    for tf in files:
        try:
            lines = open(tf, errors="ignore").read().splitlines()
        except Exception:
            continue
        for ln in lines:
            m = TXT_LINE.match(ln)
            if m:
                name, cls = m.group(1), m.group(2).strip()
                v = [float(m.group(i)) for i in range(3, 7)]
            else:
                m = TXT_LINE5.match(ln)
                if not m:
                    continue
                name, cls = None, m.group(1)
                v = [float(m.group(i)) for i in range(2, 6)]
            p = idx.resolve(name, tf) if name else None
            if p is None:   # annotation file named after the image (one txt per image)
                p = idx.resolve(os.path.splitext(os.path.basename(tf))[0], tf)
            if p is None:
                continue
            if p not in per:
                s1, sub1 = infer_split(tf, root); s2, sub2 = infer_split(p, root)
                per[p] = dict(img=os.path.splitext(os.path.basename(p))[0], path=p, split=s1 or s2,
                              subsets=sorted(set(sub1 + sub2)), objs=[], w=None, h=None, src="txt")
            per[p]["objs"].append((cls, *v))
    # images that have an (empty) per-image annotation file are still annotated negatives/positives
    return list(per.values())


def yolo_class_names(anns):
    for f in [a for a in anns if a.lower().endswith((".yaml", ".yml"))]:
        try:
            import yaml
            d = yaml.safe_load(open(f))
            n = d.get("names") if isinstance(d, dict) else None
            if isinstance(n, dict):
                return [n[k] for k in sorted(n)]
            if isinstance(n, list):
                return n
        except Exception:
            pass
    for f in [a for a in anns if os.path.basename(a).lower() in ("classes.txt", "obj.names") or a.lower().endswith(".names")]:
        return [l.strip() for l in open(f) if l.strip()]
    return None


def parse_yolo(files, idx, root, names=None):
    out = []
    for tf in files:
        p = idx.resolve(os.path.splitext(os.path.basename(tf))[0], tf.replace("labels", "images"))
        if p is None:
            continue
        rows = []
        for ln in open(tf, errors="ignore"):
            t = ln.split()
            if len(t) >= 5 and all(_safe_float(x) is not None for x in t[:5]):
                rows.append([float(x) for x in t[:5]])
        s1, sub1 = infer_split(tf, root); s2, sub2 = infer_split(p, root)
        out.append(dict(img=os.path.splitext(os.path.basename(p))[0], path=p, split=s1 or s2,
                        subsets=sorted(set(sub1 + sub2)), objs=[], w=None, h=None, src="yolo", _yolo=rows,
                        _names=names))
    return out


def parse_csv(files, idx, root):
    out = {}
    for cf in files:
        try:
            df = pd.read_csv(cf)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        def pick(*ks):
            for k in ks:
                if k in cols:
                    return cols[k]
        fc, cc = pick("filename", "file_name", "image", "img", "image_name", "path"), pick("class", "label", "category", "name")
        bx = [pick("xmin", "x1", "left"), pick("ymin", "y1", "top"), pick("xmax", "x2", "right"), pick("ymax", "y2", "bottom")]
        if fc is None or cc is None or None in bx:
            continue
        for _, r in df.iterrows():
            p = idx.resolve(str(r[fc]), cf)
            if p is None:
                continue
            if p not in out:
                s1, sub1 = infer_split(cf, root); s2, sub2 = infer_split(p, root)
                out[p] = dict(img=os.path.splitext(os.path.basename(p))[0], path=p, split=s1 or s2,
                              subsets=sorted(set(sub1 + sub2)), objs=[], w=None, h=None, src="csv")
            out[p]["objs"].append((str(r[cc]), *[float(r[b]) for b in bx]))
    return list(out.values())


def classify_ann_files(anns, root):
    """Sort annotation files by format using cheap content sniffing."""
    kinds = defaultdict(list)
    for a in anns:
        ext = os.path.splitext(a)[1].lower()
        if ext == ".json":
            try:
                if os.path.getsize(a) > 2e9:
                    continue
                with open(a) as f:
                    head = f.read(4096)
                if '"images"' in head or '"annotations"' in head or '"categories"' in head or os.path.getsize(a) > 4096:
                    kinds["coco"].append(a)
            except Exception:
                pass
        elif ext == ".xml":
            kinds["voc"].append(a)
        elif ext == ".csv":
            kinds["csv"].append(a)
        elif ext == ".txt":
            try:
                with open(a, errors="ignore") as f:
                    lines = [l.strip() for l in itertools.islice(f, 20) if l.strip()]
            except Exception:
                continue
            if not lines:
                kinds["txt_empty"].append(a); continue
            if any(TXT_LINE.match(l) or TXT_LINE5.match(l) for l in lines):
                kinds["txt"].append(a)
            elif all(len(l.split()) >= 5 and all(_safe_float(x) is not None for x in l.split()[:5]) for l in lines):
                kinds["yolo"].append(a)
            elif all(len(l.split()) == 1 for l in lines):
                kinds["idlist"].append(a)
    # cheap confirmation for voc (must contain <object> or <annotation>)
    kinds["voc"] = [x for x in kinds["voc"] if "<annotation" in open(x, errors="ignore").read(400) or
                    "<object" in open(x, errors="ignore").read()]
    return kinds


def apply_idlists(recs, idlists, root):
    """ImageSets-style id lists and OPIXray test_knife-{1,2,3}.txt.
    - occlusion lists tag OL1/OL2/OL3 (these images belong to the official test set)
    - split lists only fill in a split that folders / annotation files did not already give
      (a mirror's own random split never overrides the official one)
    - renamed mirror files (e.g. test_034894_0001.jpg) are matched through their numeric id (034894)."""
    by_stem = {r["img"].lower(): r for r in recs}
    core = defaultdict(list)
    for r in recs:
        for d in re.findall(r"\d{5,}", r["img"]):
            core[d].append(r)
    by_core = {d: v[0] for d, v in core.items() if len(v) == 1}
    used = []
    for lf in idlists:
        ids = [os.path.splitext(os.path.basename(l.strip().split()[0]))[0].lower()
               for l in open(lf, errors="ignore") if l.strip()]
        rs = [by_stem.get(i) or by_core.get(i) for i in ids]
        rs = [r for r in rs if r is not None]
        if len(rs) < max(5, 0.5 * len(ids)):
            continue
        nm = os.path.basename(lf).lower()
        s, _ = infer_split(lf, root)
        ol = re.search(r"test[\w]*?[-_](\d)\.txt$", nm)
        n_set = 0
        for r in rs:
            if ol:
                r["subsets"] = sorted(set(r["subsets"] + [f"OL{ol.group(1)}"]))
            elif r["split"] is None and (s or "trainval" in nm):
                r["split"] = "train" if "trainval" in nm else s; n_set += 1
        used.append(nm if (ol or n_set) else f"{nm} (ignored: official split kept)")
    return used


def read_sizes(recs):
    todo = [r for r in recs if not r.get("w") or not r.get("h")]
    def f(r):
        try:
            with Image.open(r["path"]) as im:
                r["w"], r["h"] = im.size
        except Exception:
            r["w"] = r["h"] = None
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(f, todo))


def load_dataset(key, roots):
    roots = roots if isinstance(roots, (list, tuple)) else [roots]
    root = roots if len(roots) > 1 else roots[0]
    imgs, anns = index_files(roots)
    idx = ImageIndex(imgs, root)
    kinds = classify_ann_files(anns, root)
    order = ["coco", "voc", "txt", "csv", "yolo"]
    fmt = FORCE_FORMAT.get(key) if key else None
    fmt = fmt or next((k for k in order if kinds.get(k)), None)
    yolo_names, yolo_note = None, None
    if fmt == "yolo":
        yolo_names = yolo_class_names(anns)
        if yolo_names is None and key in YOLO_DEFAULT_NAMES:
            yolo_names = YOLO_DEFAULT_NAMES[key]
            yolo_note = f"no data.yaml/classes.txt found -> ASSUMED official class order {yolo_names}. VERIFY!"
    parsers = dict(coco=parse_coco, voc=parse_voc, txt=parse_txt, csv=parse_csv,
                   yolo=lambda f, i, r: parse_yolo(f, i, r, yolo_names))
    recs = parsers[fmt](kinds[fmt], idx, root) if fmt in parsers else []
    if fmt == "coco" and not recs:
        for alt in ["voc", "txt", "csv", "yolo"]:
            if kinds.get(alt) and alt != "yolo":
                fmt, recs = alt, parsers[alt](kinds[alt], idx, root)
                if recs:
                    break
    # dedupe: same file listed twice, or the same image present in two attached mirrors (same file name)
    seen, by_base = {}, {}
    for r in recs:
        b = os.path.basename(r["path"]).lower()
        if r["path"] in seen or (b in by_base and by_base[b] != r["path"]):
            t = seen.get(r["path"]) or seen[by_base[b]]
            t["objs"] += [o for o in r["objs"] if o not in t["objs"]]
            t["subsets"] = sorted(set(t["subsets"] + r["subsets"]))
        else:
            seen[r["path"]] = r; by_base[b] = r["path"]
    recs = list(seen.values())
    used_lists = apply_idlists(recs, kinds.get("idlist", []), root)
    read_sizes(recs)
    for r in recs:   # YOLO normalized -> absolute
        if "_yolo" in r:
            nm = r.pop("_names") or []
            for c, cx, cy, bw, bh in r.pop("_yolo"):
                W, H = r["w"] or 1, r["h"] or 1
                r["objs"].append((nm[int(c)] if int(c) < len(nm) else f"class{int(c)}",
                                  (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    # negatives (clean bags) = images never mentioned by any annotation
    annotated = {os.path.basename(r["path"]).lower() for r in recs}
    negs, nb = [], set()
    if key in NEG_DATASETS:
        for p in imgs:
            b = os.path.basename(p).lower()
            if b not in annotated and b not in nb:
                negs.append(p); nb.add(b)
    example = None
    if kinds.get(fmt):
        try:
            example = open(kinds[fmt][0], errors="ignore").read(600)
        except Exception:
            pass
    classes = Counter(canon_name(o[0]) for r in recs for o in r["objs"])
    info = dict(roots=roots, format=fmt, n_image_files=len(imgs), n_ann_files={k: len(v) for k, v in kinds.items()},
                n_annotated_images=len(recs), n_unannotated_images=len(negs), idlists_used=used_lists,
                classes=dict(classes.most_common()), yolo_note=yolo_note, example_annotation=example,
                archives=[a for r in roots for a in list_archives(r, 5)])
    return recs, negs, info


def canon_name(n):
    return re.sub(r"[^a-z0-9]", "", str(n).lower())


def fingerprint(classes):
    got = {canon(c) for c in classes} if "canon" in globals() else set()
    raw = {canon_name(c) for c in classes}
    for k, sig in FINGERPRINT.items():
        sig_raw = {canon_name(x) for x in sig}
        hits = len(sig_raw & raw) + len(sig & got)
        if k == "sixray":
            if len(sig_raw & raw) >= 4 and not (raw - sig_raw - {"scissor"}):
                return k
        elif hits >= 2:
            return k
    return None


# canonical-name helper is defined in the ontology cell; a light version is enough for fingerprinting here
_CANON_LITE = {"scissor": "scissors", "multitoolknife": "multitool_knife", "foldingknife": "folding_knife",
               "straightknife": "straight_knife", "utilityknife": "utility_knife", "handcuff": "handcuffs"}


def canon(n):
    k = canon_name(n)
    return _CANON_LITE.get(k, re.sub(r"[^a-z0-9]+", "_", str(n).lower()).strip("_"))


def report(k, d, secs):
    print(f"\n[{k}] format={d['format']}  image files={d['n_image_files']}  annotated={d['n_annotated_images']}  "
          f"unannotated={d['n_unannotated_images']}  ann-files={d['n_ann_files']}  id-lists={d['idlists_used']}  ({secs:.0f}s)")
    print(f"   classes: {d['classes']}")
    if d.get("yolo_note"):
        print("   ⚠️ ", d["yolo_note"])
    if d["example_annotation"]:
        print("   example annotation snippet:\n   " + d["example_annotation"][:300].replace("\n", "\n   "))


if NEED_DISCOVERY:
    resume_from_input()
    GROUPS, UNNAMED = find_dataset_roots()
    print("Name-matched Kaggle inputs:", json.dumps(GROUPS, indent=1))
    print("Other inputs (identified by content):", UNNAMED)

    RAW, RAW_NEG, DISCOVERY = {}, {}, {}
    for k, roots in GROUPS.items():
        for r in roots:
            print("\n" + "=" * 90 + f"\n{k.upper()}"); print_tree(r)
        t0 = time.time()
        roots = GROUPS[k] = prepare_multipart(k, roots)
        RAW[k], RAW_NEG[k], DISCOVERY[k] = load_dataset(k, roots)
        report(k, DISCOVERY[k], time.time() - t0)

    for r in UNNAMED:   # e.g. "xray-contraband-detection" -> fingerprinted as PIDray
        t0 = time.time()
        recs, negs, info = load_dataset(None, [r])
        k = fingerprint(info["classes"]) if recs else None
        if k is None:
            print(f"\n[unidentified input] {r}: {len(recs)} annotated images, classes={list(info['classes'])[:10]} -> ignored "
                  f"(set DATASET_ROOTS if this is one of {DS_KEYS})")
            continue
        if RAW.get(k):
            print(f"\n[{k}] also found in {r} (content match) -> ignored because {GROUPS[k]} is already used")
            continue
        print("\n" + "=" * 90 + f"\n{k.upper()}  (identified by class names)"); print_tree(r)
        if k in NEG_DATASETS:
            recs, negs, info = load_dataset(k, [r])
        RAW[k], RAW_NEG[k], DISCOVERY[k] = recs, negs, info
        GROUPS[k] = [r]
        report(k, info, time.time() - t0)

    for k in list(RAW):
        if not RAW[k]:
            print(f"   ❌ no annotations parsed for {k} ({GROUPS[k]}). "
                  + ("It only contains archives -> attach an extracted mirror or set EXTRACT_ARCHIVES=True. " if DISCOVERY[k]["archives"] else "")
                  + f"Set FORCE_FORMAT['{k}'] / DATASET_ROOTS['{k}'] and re-run.")
            RAW.pop(k); RAW_NEG.pop(k)
    DATASET_ROOTS_FOUND = {k: GROUPS[k] for k in RAW}
    missing = [k for k in DS_KEYS if k not in RAW]
    if missing:
        print("\n⚠️  not available (runs that need them are skipped):", missing)
    if "opixray" in RAW and not any("OL" in s for r in RAW["opixray"] for s in r["subsets"]):
        print("⚠️  OPIXray occlusion lists test_knife-{1,2,3}.txt not found -> P6 (OL1/OL2/OL3, ORD) will be empty. "
              "Upload these 4 small txt files as a Kaggle dataset whose name contains 'opixray' (e.g. opixray-ol-lists).")
    jdump(DISCOVERY, WORK / "dataset_discovery.json")
else:
    print('pack found -> discovery skipped (records, audit and verification are loaded from the pack)')
