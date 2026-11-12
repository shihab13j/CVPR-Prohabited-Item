# XGuard — Unknown-Threat Discovery under Compound Distribution Shift in X-ray Security Inspection

**Colab Pro (A100) notebook.** Data comes from Kaggle **once**, on a cheap CPU runtime, and is stored on your Google Drive as a compact pack. GPU sessions never download datasets.

## How to run (3 steps)

| Step | Runtime | Set in the config cell | What happens | Time / cost |
|---|---|---|---|---|
| **1. prepare** (once) | **CPU** (Runtime → Change runtime type → CPU) | `MODE = "prepare"` | Kaggle API downloads the 5 mirrors in parallel to local SSD → discovery, verification against official dataset sizes, lineage & leakage audit → compact pack (images at 800 px, JPEG q95, boxes in original coordinates) written to `MyDrive/xguard/pack_v2` | ~30–60 min on CPU (≈ 0 GPU units) |
| **2. test** | **A100** | `MODE = "test"` | Copies the pack to local SSD, runs the correctness unit tests, the reproducibility/resume test and **every** model variant, metric, table, figure and release gate on **real data** with a few iterations, then prints the measured **time and compute units per tier** | ~30–45 min |
| **3. full** | **A100** | `MODE = "full"`, `TIERS = [1]` (then `[2]`, `[3]`) | The real experiments. Everything is saved to Drive after every epoch; if Colab disconnects, simply *Run all* again — finished runs are skipped (verified by fingerprints) | see the estimate printed by step 2 |

**Before step 1:** kaggle.com → Settings → API → *Create New Token*. In Colab click the 🔑 *Secrets* icon and add `KAGGLE_USERNAME` and `KAGGLE_KEY` (enable notebook access). Optional: put OPIXray's `test_knife.txt`, `test_knife-1.txt`, `test_knife-2.txt`, `test_knife-3.txt` into `MyDrive/xguard/opixray_ol/` (this enables the OL1–OL3 occlusion table).

| Key | Kaggle dataset | Verified content (real Kaggle run) |
|---|---|---|
| sixray | `khanhbtq99/sixray` | Roboflow export: train 11,638 / valid 1,662 / test 831 files with **augmented copies** (`P00031_jpg.rf.<hash>.jpg`) and **5 classes (no hammer)** → collapsed to one copy per original id, ids never shared across splits |
| sixray-p1 | `witsarutsarai/sixray-p1` (file `JPEGImage.tar.gz00` only, ~4 GB) | first part of the official SIXray archive → clean bags (N*.jpg) for false-alarm metrics; the other 15 parts are not needed |
| pidray | `banan41k/xray-contraband-detection` | PIDray: 47,674 PNG + official COCO jsons; test = 9,482 easy + 3,733 hard + 5,005 hidden = **18,220 (matches the paper)**; the mirror's own random `splits/*.txt` are ignored |
| hixray | `newashish/hixray` | official 36,295 train / 9,069 test, per-image TXT |
| opixray | `alirzaei/opixray-yolo` | YOLO; train 7,109; official test 1,776 cut into val 888 + test 888 → merged back |

**Honesty rules built into the code:** no target data is used for training, thresholds or checkpoint selection (source validation only); exact / harmonized / hierarchical AP are reported separately; leakage exclusions are frozen before any evaluation; metrics that do not apply are marked `n/a` with a reason; stale results are never reused (fingerprints); zero-shot "oracle vocabulary" baselines are labelled as references; methods without public code (OVXD, STING-BEE, CHR, DOAM, LIM, SDANet) are not re-implemented — published numbers can be merged via `C.EXTERNAL_RESULTS` and are marked *(reported)*. The model is a two-stage R50-FPN detector (not DINO/DETR) for compute reasons — state this in the paper.
