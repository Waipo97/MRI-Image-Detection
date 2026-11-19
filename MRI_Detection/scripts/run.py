"""Run the pipeline from the command line (also what the Colab notebook calls):
    python scripts/run.py prepare
    python scripts/run.py baseline [--quick]
    python scripts/run.py improved [--quick]
    python scripts/run.py evaluate [--quick]
    python scripts/run.py all      [--quick]     # prepare -> baseline -> improved -> evaluate

Paths: defaults are inside the repo. Override with environment variables (used on Colab):
    MRI_WORK_DIR  local fast disk for data      MRI_RUNS_DIR  persistent folder (Drive) for runs
    MRI_RAW_ZIP   path to archive.zip           MRI_WEIGHTS_DIR  where COCO yolov8m.pt is cached
Training is resumable: re-running a step continues from last.pt or skips a finished run.
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
import mri_detection as md
from mri_detection import data, train, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prepare", "baseline", "improved", "evaluate", "all"])
ap.add_argument("--quick", action="store_true", help="3 epochs, 320 px, 25%% of train (sanity run)")
ap.add_argument("--epochs", type=int, help="override train.epochs")
a = ap.parse_args()

cfg = md.load_config(quick=a.quick)
if a.epochs:
    cfg["train"]["epochs"] = a.epochs
P = md.get_paths(cfg)
P["runs"].mkdir(parents=True, exist_ok=True); P["manifests"].mkdir(parents=True, exist_ok=True)


def prepare():
    root = data.extract_zip(P["raw_zip"], P["raw_dir"])
    df, _, names = data.clean_dataset(data.scan_dataset(root))
    s = cfg["dataset"]["split"]
    df = data.split_dataset(df, (s["train"], s["val"], s["test"]), cfg["seed"])
    rel = lambda p: str(Path(p).relative_to(root))
    export = df.assign(image=df.image.map(rel))[["image", "folder", "n_boxes", "issue", "split"]]
    export[export.issue.notna()].to_csv(P["manifests"] / "excluded_images.csv", index=False)
    export[export.split.notna()].to_csv(P["manifests"] / "split_manifest.csv", index=False)
    print(data.build_yolo_dataset(df, names, P["yolo_dir"]))
    print(data.split_summary(df, names))


def class_names():
    return list(yaml.safe_load(open(P["data_yaml"]))["names"].values())


def evaluate_both():
    dev = md.config.resolve_device(cfg["train"]["device"])
    res = evaluate.compare(P["runs"], P["data_yaml"], cfg["train"]["imgsz"], cfg["train"]["batch"], dev)
    for split, (tab, pc) in res.items():
        print(f"\n===== {split.upper()} ====="); print(tab.round(4)); print(pc.round(4))


steps = ["prepare", "baseline", "improved", "evaluate"] if a.step == "all" else [a.step]
for st in steps:
    print(f"\n########## {st} ##########")
    if st == "prepare":
        prepare()
    elif st == "baseline":
        print(train.train_baseline(cfg, P["data_yaml"], P["runs"]))
    elif st == "improved":
        print(train.train_improved(cfg, P["data_yaml"], P["runs"], nc=len(class_names())))
    else:
        evaluate_both()
