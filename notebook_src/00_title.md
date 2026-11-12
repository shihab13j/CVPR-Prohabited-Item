# XGuard — Unknown-Threat Discovery under Compound Distribution Shift in X-ray Security Inspection

One notebook, two platforms: **Tier 1 on Colab Pro (A100)** and **Tiers 2–3 on Kaggle (free GPU)**. Everything that does not need a GPU happens on a CPU runtime, so no GPU time is spent on downloads, audits, CLIP loading or tables.

## Colab Pro (Tier 1: core with 3 seeds + component ablation)
| Step | Runtime | Config | What happens |
|---|---|---|---|
| 1 | **CPU** | `MODE="prepare"` | Downloads the Kaggle mirrors once (add the Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` first). Then it verifies the data, runs the leakage audit, and pre-computes the CLIP text embeddings and the COCO weights. Output is a compact pack in `MyDrive/xguard/pack_v2`. |
| 2 | **A100** | `MODE="test"` | All tier-1 code paths on real data with a few iterations, the unit tests and the resume test. It prints the measured **GPU-hours and compute units** for tier 1, then **releases the runtime by itself**. |
| 3 | **A100** | `MODE="full"` | Tier 1. SIXray is unpacked first and training starts right away; the other test sets unpack in the background. After each epoch everything is saved to Drive. When the selected runs finish, or if the GPU is idle for 20 minutes, the runtime is **released automatically**. After a disconnect, choose *Run all* again; finished runs are skipped. |
| 4 | **CPU** | `MODE="report"` | Put the Kaggle `xguard_results_full.zip` into `MyDrive/xguard/kaggle_results/`. This step merges everything into Tables T0–T14, Figures 1–8 and the release gates, without any GPU cost. |

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
