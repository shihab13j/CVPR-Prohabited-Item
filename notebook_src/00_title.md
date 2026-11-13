# XGuard — Unknown-Threat Discovery under Compound Distribution Shift in X-ray Security Inspection

One notebook, two platforms: **Tier 1 on Colab Pro (A100)** and **Tiers 2–3 on Kaggle (free GPU)**. Everything that does not need a GPU happens on a CPU runtime, so no GPU time is spent on downloads, audits, CLIP loading or tables.

## Colab Pro (Tier 1: core with 3 seeds + component ablation): `MODE = "auto"`, so you never edit the code
You only choose the **runtime type** and click **Run all**. The notebook decides which step to run:

| You open | Data pack on Drive? | It runs |
|---|---|---|
| **CPU** runtime | no | **prepare**: Kaggle download → verification → leakage audit → pack on Drive (add the Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` first) |
| **A100** runtime | yes, test not passed yet | **test**: every tier-1 code path on real data, plus the GPU-hours / compute-unit estimate. The runtime is released at the end. |
| **A100** runtime | yes, test passed | **full**: the tier-1 experiments. Resumable: after a disconnect, just click Run all again. The runtime is released when the work is done or after 20 idle minutes. |
| **CPU** runtime | yes | **report**: merges Colab + Kaggle results into tables, figures and gates |
| **A100** runtime | **no** | stops at once and **releases the GPU** (prepare must run on a CPU runtime) |

If an error ever stops the notebook: **Runtime → Disconnect and delete runtime** releases the GPU immediately.

## Kaggle (Tiers 2–3: LOCO first, then the supporting ablations, baselines, zero-shot, in-domain, LODO, encoders, YOLO / RT-DETR)
1. **CPU session.** Add Input: `khanhbtq99/sixray`, `witsarutsarai/sixray-p1`, `banan41k/xray-contraband-detection`, `newashish/hixray`, `alirzaei/opixray-yolo`, and optionally your OPIXray OL lists. Set `MODE="prepare"` and choose *Save & Run All*. The pack is written to the output.
2. **GPU session** (Accelerator **GPU T4×2/P100**, Internet **on**). Add Input: the output of step 1. Set `MODE="test"`, then `MODE="full"`, using *Save & Run All* so the run continues when the browser is closed. Each session lasts at most 12 h; add the previous output as input and run again to continue.
3. Download `xguard_results_full.zip` and put it into `MyDrive/xguard/kaggle_results/` for step 4 above.

Kaggle uses batch 4 with gradient accumulation 2, which gives the same effective batch of 8 as Colab, so the results are comparable.

| Key | Kaggle dataset | Verified content (real Kaggle run) |
|---|---|---|
| sixray | `khanhbtq99/sixray` | Roboflow export with augmented copies and **5 classes (no hammer)**; collapsed to one copy per original image ID, IDs never shared across splits |
| sixray-p1 | `witsarutsarai/sixray-p1` (first part only on Colab) | clean bags for the false-alarm metrics |
| pidray | `banan41k/xray-contraband-detection` | PIDray; test 18,220 = 9,482 easy + 3,733 hard + 5,005 hidden (matches the paper) |
| hixray | `newashish/hixray` | official 36,295 / 9,069 |
| opixray | `alirzaei/opixray-yolo` | official test of 1,776 split into val + test, merged back here |

**Honesty rules in the code:**
- No target data is used for training, thresholds or checkpoint choice.
- Exact, harmonized and hierarchical AP are reported separately.
- Leakage exclusions are frozen before any evaluation.
- Metrics that do not apply show `n/a` with a reason.
- Fingerprints prevent stale results from being reused.
- Oracle-vocabulary zero-shot runs are labelled as references only.
- Methods without public code are not re-implemented.
- The model is a two-stage R50-FPN, not DETR, for compute reasons; state this in the paper.
