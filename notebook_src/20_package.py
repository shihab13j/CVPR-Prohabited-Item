# ==============================================================================
# 19. PACKAGE deliverables (tables, figures, audit, manifests, metrics, gates; no weights / predictions)
# ==============================================================================
if MODE == "prepare":
    raise StopExecution()   # prepare mode ends after the pack cell
pkg = CACHE / f"package_{MODE}"
if pkg.exists():
    shutil.rmtree(pkg)
pkg.mkdir(parents=True)
for f in [f"config_{MODE}.json", "dataset_stats.json", "class_mapping.json", f"gates_{MODE}.json"]:
    if (WORK / f).exists():
        shutil.copy(WORK / f, pkg / f)
for d in ["audit", "logs"]:
    if (WORK / d).exists():
        shutil.copytree(WORK / d, pkg / d, ignore=shutil.ignore_patterns("_repro_*"))
shutil.copytree(TAB, pkg / "tables"); shutil.copytree(FIG, pkg / "figures")
for rd in all_runs():
    if rd.is_dir() and not rd.name.startswith("_"):
        files = list(rd.glob("*.json")) + list(rd.glob("*.txt")) + list(rd.glob("*.csv")) + list(rd.glob("*.npz")) + list(rd.glob("DONE"))
        if files:
            (pkg / "runs" / rd.name).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, pkg / "runs" / rd.name / f.name)
zp = shutil.make_archive(str(WORK / f"xguard_results_{MODE}"), "zip", pkg)
print("results archive:", zp, f"({os.path.getsize(zp) / 1e6:.1f} MB)")
todo = [s["name"] for s in SELECTED if not (run_path(s["name"]) / "DONE").exists()]
print(f"session time used: {elapsed_h():.2f} h | selected runs still to do: {len(todo)}")
if todo:
    print("➡  re-run the notebook (same MODE / TIERS) to continue; finished runs are kept on Drive and skipped.")
if PLATFORM == "kaggle" and MODE == "full":
    print("Kaggle: download", zp, "and put it into MyDrive/xguard/kaggle_results/ -> Colab MODE='report' merges everything.")
if PLATFORM == "colab" and MODE == "test":
    release_runtime("test finished — read the gates and the time estimate above, if the test passed, just Connect a GPU runtime and Run all again: MODE 'auto' starts the full experiments")
if PLATFORM == "colab" and MODE == "full":
    release_runtime("all planned runs, tables, figures and the results zip are finished"
                    + (f" ({len(todo)} runs ended with an error: see the run status table)" if todo else ""))
