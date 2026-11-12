"""Tiny synthetic copy of the REAL Kaggle mirrors used by XGuard (layout verified from a real Kaggle run):

  sixray/      khanhbtq99/sixray  : Roboflow export sixray_v3/{train,valid,test}/{images,labels}, names P00031_jpg.rf.<hash>.jpg,
                                    augmented copies of the same id, some empty label files, data.yaml, 5 classes (no hammer)
  sixray-p1/   witsarutsarai/sixray-p1 : ONLY JPEGImage.tar.gz00 = the first chunk of a split tar.gz (clean bags N*.jpg)
  pidray/      banan41k/xray-contraband-detection : images/xray_XXXXX.png, annotations/xray_{train,test_easy,test_hard,test_hidden}.json,
                                    splits/{train,val,test}_images.txt (the mirror's own RANDOM split), dataset_split.csv
  hixray/      newashish/hixray   : {train,test}/{*_image,*_annotation}, lines "img.jpg Class x1 y1 x2 y2"
  opixray/     alirzaei/opixray-yolo : OPIXray/{images,labels}/{train,val,test}/<split>_<id>_0001.jpg + dataset.yaml
                                    (official test cut into val + test)
  opixray-ol/  test_knife{,-1,-2,-3}.txt with official numeric ids (optional user upload)

  python tools/make_synthetic_datasets.py OUT_DIR
"""
import io, json, random, sys, tarfile, hashlib
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(0)
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_raw")
PAL = {}


def img_with(objs, W=220, H=170):
    im = Image.new("RGB", (W, H), (random.randint(200, 255), random.randint(200, 255), random.randint(180, 240)))
    d = ImageDraw.Draw(im)
    for _ in range(4):
        x, y = random.randint(0, W - 30), random.randint(0, H - 30)
        d.rectangle([x, y, x + random.randint(10, 40), y + random.randint(10, 40)], fill=(random.randint(80, 200),) * 3)
    boxes = []
    for c in objs:
        col = PAL.setdefault(c, tuple(random.randint(0, 150) for _ in range(3)))
        w, h = random.randint(25, 70), random.randint(20, 60)
        x, y = random.randint(0, W - w - 1), random.randint(0, H - h - 1)
        d.rectangle([x, y, x + w, y + h], fill=col)
        boxes.append((c, x, y, x + w, y + h))
    return im, boxes


def yolo_line(ci, b, W, H):
    _, x1, y1, x2, y2 = b
    return f"{ci} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"


def sixray_roboflow():
    cls = ["gun", "knife", "pliers", "scissors", "wrench"]
    base = ROOT / "sixray" / "sixray_v3"
    (base).mkdir(parents=True)
    (base / "README.dataset.txt").write_text("Roboflow export of SIXray\n")
    (base / "README.roboflow.txt").write_text("augmentations: flip, brightness\n")
    (base / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 5\nnames: " + json.dumps(cls) + "\n")
    pid = 1
    for split, n, copies in [("train", 40, 3), ("valid", 8, 1), ("test", 12, 1)]:
        (base / split / "images").mkdir(parents=True); (base / split / "labels").mkdir(parents=True)
        for _ in range(n):
            name = f"P{pid:05d}"; pid += 1
            objs = random.sample(cls, random.randint(1, 2))
            for c in range(copies):
                im, bx = img_with(objs)
                h = hashlib.md5(f"{name}{c}".encode()).hexdigest()
                fn = f"{name}_jpg.rf.{h}"
                im.save(base / split / "images" / f"{fn}.jpg")
                lab = "" if (split == "train" and c == 2 and pid % 5 == 0) else "".join(yolo_line(cls.index(b[0]), b, im.width, im.height) for b in bx)
                (base / split / "labels" / f"{fn}.txt").write_text(lab)
    # one original id leaking into two splits (the notebook must keep it in test only)
    leak = sorted((base / "test" / "images").iterdir())[0]
    lid = leak.name.split("_")[0]
    im = Image.open(leak); im.save(base / "train" / "images" / f"{lid}_jpg.rf.leak.jpg")
    (base / "train" / "labels" / f"{lid}_jpg.rf.leak.txt").write_text((base / "test" / "labels" / (leak.stem + ".txt")).read_text())


def sixray_part00():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for i in range(60):
            name = f"N{i:07d}" if i < 40 else f"P{i:05d}"
            im, _ = img_with([])
            b = io.BytesIO(); im.save(b, "JPEG"); data = b.getvalue()
            ti = tarfile.TarInfo(f"JPEGImage/{name}.jpg"); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    data = buf.getvalue()
    (ROOT / "sixray-p1").mkdir(parents=True)
    (ROOT / "sixray-p1" / "JPEGImage.tar.gz00").write_bytes(data[: int(len(data) * 0.8)])   # truncated: first part only


def pidray_mirror():
    pid = ["Baton", "Pliers", "Hammer", "Powerbank", "Scissors", "Wrench", "Gun", "Bullet", "Sprayer", "HandCuffs", "Knife", "Lighter"]
    r = ROOT / "pidray"; (r / "images").mkdir(parents=True); (r / "annotations").mkdir(); (r / "splits").mkdir()
    k, allimgs = 0, []
    for jname, n_img in [("xray_train", 40), ("xray_test_easy", 12), ("xray_test_hard", 10), ("xray_test_hidden", 10)]:
        images, anns = [], []
        for _ in range(n_img):
            fn = f"xray_{k:05d}.png"; k += 1
            im, bx = img_with(random.sample(pid, random.randint(1, 3))); im.save(r / "images" / fn); allimgs.append(fn)
            images.append(dict(id=len(images), file_name=fn, width=im.width, height=im.height))
            for c, x1, y1, x2, y2 in bx:
                anns.append(dict(id=len(anns), image_id=len(images) - 1, category_id=pid.index(c) + 1, bbox=[x1, y1, x2 - x1, y2 - y1],
                                 area=(x2 - x1) * (y2 - y1), iscrowd=0))
        json.dump(dict(info="spytensor created", images=images, annotations=anns, categories=[dict(id=i + 1, name=c) for i, c in enumerate(pid)]),
                  open(r / "annotations" / f"{jname}.json", "w"))
    random.shuffle(allimgs)
    a, b = int(0.8 * len(allimgs)), int(0.9 * len(allimgs))
    for nm, part in [("train", allimgs[:a]), ("val", allimgs[a:b]), ("test", allimgs[b:])]:
        (r / "splits" / f"{nm}_images.txt").write_text("\n".join(part) + "\n")
    (r / "dataset_split.csv").write_text("image,split\n" + "".join(f"{x},train\n" for x in allimgs[:a]))


def hixray_official():
    cls = ["Portable_Charger_1", "Portable_Charger_2", "Mobile_Phone", "Laptop", "Tablet", "Cosmetic", "Water", "Nonmetallic_Lighter"]
    for split, n in [("train", 36), ("test", 16)]:
        (ROOT / "hixray" / split / f"{split}_image").mkdir(parents=True)
        (ROOT / "hixray" / split / f"{split}_annotation").mkdir(parents=True)
        for i in range(n):
            name = f"{random.randint(10 ** 14, 10 ** 15 - 1)}.jpg"
            im, bx = img_with(random.sample(cls, random.randint(1, 3)))
            im.save(ROOT / "hixray" / split / f"{split}_image" / name)
            (ROOT / "hixray" / split / f"{split}_annotation" / name.replace(".jpg", ".txt")).write_text(
                "".join(f"{name} {c} {x1} {y1} {x2} {y2}\n" for c, x1, y1, x2, y2 in bx))


def opixray_yolo():
    ocls = ["Folding_Knife", "Straight_Knife", "Scissor", "Utility_Knife", "Multi-tool_Knife"]
    o = ROOT / "opixray" / "OPIXray"; test_ids = []
    for split, n_img, start in [("train", 36, 9000), ("val", 9, 10466), ("test", 9, 34894)]:
        (o / "images" / split).mkdir(parents=True, exist_ok=True); (o / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n_img):
            oid = f"{start + i:06d}"; name = f"{split}_{oid}_0001"
            im, bx = img_with([random.choice(ocls)]); im.save(o / "images" / split / f"{name}.jpg")
            (o / "labels" / split / f"{name}.txt").write_text("".join(yolo_line(ocls.index(b[0]), b, im.width, im.height) for b in bx))
            if split != "train":
                test_ids.append(oid)
    (o / "dataset.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {json.dumps(ocls)}\n")
    (ROOT / "opixray-ol").mkdir(parents=True)
    (ROOT / "opixray-ol" / "test_knife.txt").write_text("\n".join(test_ids) + "\n")
    for j in (1, 2, 3):
        (ROOT / "opixray-ol" / f"test_knife-{j}.txt").write_text("\n".join(test_ids[j - 1::3]) + "\n")


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    sixray_roboflow(); sixray_part00(); pidray_mirror(); hixray_official(); opixray_yolo()
    print("synthetic copy of the Kaggle mirrors written to", ROOT)
