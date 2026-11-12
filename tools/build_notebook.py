"""Assemble notebook_src/*.py|*.md into XGuard_Colab.ipynb (one cell per source file)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook_src"
HEADERS = {
    "01_config.py": "## 0 · Configuration\nSet `MODE` (`prepare` → `test` → `full`) and `TIERS`. Nothing else needs editing.",
    "02_env.py": "## 1 · Environment\nDrive mount (Colab), GPU check (stops if there is no GPU), bf16/TF32 on A100, helpers.",
    "03_stage.py": "## 2 · Data staging\n`prepare`: Kaggle API → local SSD (parallel downloads). `test`/`full`: copy the pack from Drive to local SSD.",
    "04_discovery.py": "## 3 · Dataset discovery (prepare only)\nFolder tree, annotation format, splits and occlusion lists of every downloaded dataset.",
    "05_ontology.py": "## 4 · Ontology\nExact / harmonized / hierarchical tracks, reviewed aliases, superclasses, attributes.",
    "06_records.py": "## 5 · Manifest & data integrity\nLineage (original image ids), official-size verification, split metadata, class presence audit.",
    "07_audit.py": "## 6 · Leakage audit\nSHA-256 exact duplicates, confirmed near-duplicates, clusters, cross-dataset overlap, frozen exclusions.",
    "08_pack.py": "## 7 · Pack\n`prepare` writes the compact pack to Drive and stops here.",
    "09_data.py": "## 8 · Data pipeline & augmentations",
    "10_model.py": "## 9 · XGuard model\nSIRA · OER · STPB · CUS · RCR, baselines, ablation variants.",
    "11_train.py": "## 10 · Training & inference\nExact resume, accumulation tail, NaN policy, source-val checkpoint selection.",
    "12_eval.py": "## 11 · Evaluation\nThree AP tracks, open-world metrics, reliability, bag-level referral, metric status.",
    "13_tests.py": "## 12 · Correctness tests\nMetric unit tests (stop on failure), CUS/ontology/CLIP integrity, reproducibility.",
    "14_protocols.py": "## 13 · Protocols & experiment registry",
    "15_run.py": "## 14 · Run\nLong cell. Resumable: re-run the notebook to continue. Test mode also prints the full-plan time / compute-unit estimate.",
    "16_tables.py": "## 15 · Tables T0–T14",
    "17_analysis.py": "## 16 · Representation evidence (SIRA validation)",
    "18_figures.py": "## 17 · Figures 1–8",
    "19_gates.py": "## 18 · Release gates",
    "20_package.py": "## 19 · Package results",
}

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"},
               "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"}}
for f in sorted(p for p in SRC.iterdir() if p.suffix in (".py", ".md")):
    txt = f.read_text().rstrip() + "\n"
    if f.suffix == ".md":
        nb.cells.append(nbf.v4.new_markdown_cell(txt))
    else:
        if f.name in HEADERS:
            nb.cells.append(nbf.v4.new_markdown_cell(HEADERS[f.name]))
        nb.cells.append(nbf.v4.new_code_cell(txt))
out = ROOT / "XGuard_Colab.ipynb"
nbf.write(nb, out)
print("wrote", out, len(nb.cells), "cells")
