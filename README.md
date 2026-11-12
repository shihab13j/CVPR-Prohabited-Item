# XGuard — Unknown-Threat Discovery under Compound Shift in X-ray Security Inspection

**Deliverable:** [`XGuard_Colab.ipynb`](XGuard_Colab.ipynb). It is built from `notebook_src/`, one file per cell. Run it on Colab Pro: `prepare` on a CPU runtime, then `test` and `full` on an A100. It also runs on Kaggle with the GPU accelerator turned **on**. Your earlier Kaggle run was on CPU, which is why one epoch took 10.5 hours.

## Workflow
1. **`MODE="prepare"` on a Colab CPU runtime.** First add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets. This step then:
   - downloads the Kaggle mirrors once
   - verifies them against the official dataset sizes
   - audits lineage and leakage
   - writes a compact pack to `MyDrive/xguard/pack_v2`
2. **`MODE="test"` on an A100.** On real data, with a few iterations per model, this runs:
   - the unit tests
   - the reproducibility and resume test
   - every model variant, metric, table, figure and release gate

   It then prints the measured time and compute-unit cost of the full plan, per tier.
3. **`MODE="full"`, `TIERS=[1]`, then `[2]` and `[3]`.** All outputs live on Drive. Run the notebook again after a disconnect and finished runs are skipped. Fingerprints stop stale results from being reused.

| Key | Kaggle dataset | Verified layout |
|---|---|---|
| sixray | khanhbtq99/sixray | Roboflow export with augmented copies and 5 classes (no hammer). It is collapsed to one copy per original image ID, and no ID is shared between splits. |
| sixray-p1 | witsarutsarai/sixray-p1, file `JPEGImage.tar.gz00` only | Clean bags used for the false-alarm metrics |
| pidray | banan41k/xray-contraband-detection | Official COCO splits; test is 18,220 images (easy 9,482, hard 3,733, hidden 5,005) |
| hixray | newashish/hixray | Official split: 36,295 train / 9,069 test |
| opixray | alirzaei/opixray-yolo | Official test (1,776) is split into val + test; the notebook merges them back. The OL1–3 lists are optional (`MyDrive/xguard/opixray_ol/`). |

## Development
```bash
python tools/build_notebook.py                         # rebuild the notebook from notebook_src/
python tools/make_synthetic_datasets.py /tmp/raw       # tiny synthetic copy of the real Kaggle layout
XGUARD_SMOKE=1 XGUARD_MODE=prepare XGUARD_PLATFORM=local XGUARD_INPUT=/tmp/raw XGUARD_WORK=/tmp/w \
  XGUARD_PACK=/tmp/pack XGUARD_PACK_DRIVE=/tmp/packdrive XGUARD_CACHE=/tmp/cache \
  jupyter nbconvert --to notebook --execute XGuard_Colab.ipynb --output /tmp/out_prepare.ipynb
# then the same command with XGUARD_MODE=test
```
