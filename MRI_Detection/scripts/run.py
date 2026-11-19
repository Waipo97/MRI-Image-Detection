"""Run the pipeline from the command line:
    python scripts/run.py prepare
    python scripts/run.py baseline [--quick]
    python scripts/run.py improved [--quick]
    python scripts/run.py evaluate [--quick]
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
import mri_detection as md
from mri_detection import data, train, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prepare", "baseline", "improved", "evaluate"])
ap.add_argument("--quick", action="store_true", help="3 epochs, 320 px, 25%% of train (CPU sanity run)")
a = ap.parse_args()

cfg = md.load_config(quick=a.quick)
RUNS = ROOT / "runs"
YOLO_DIR = ROOT / cfg["dataset"]["yolo_dir"]
DATA_YAML = YOLO_DIR / "data.yaml"

if a.step == "prepare":
    root = data.extract_zip(ROOT / cfg["dataset"]["raw_zip"], ROOT / cfg["dataset"]["raw_dir"])
    df, _, names = data.clean_dataset(data.scan_dataset(root))
    s = cfg["dataset"]["split"]
    df = data.split_dataset(df, (s["train"], s["val"], s["test"]), cfg["seed"])
    print(data.build_yolo_dataset(df, names, YOLO_DIR)); print(data.split_summary(df, names))
else:
    names = list(yaml.safe_load(open(DATA_YAML))["names"].values())
    if a.step == "baseline":
        print(train.train_baseline(cfg, DATA_YAML, RUNS))
    elif a.step == "improved":
        print(train.train_improved(cfg, DATA_YAML, RUNS, nc=len(names)))
    else:
        dev = md.config.resolve_device(cfg["train"]["device"])
        for n, d in [("baseline", "baseline_yolov8m"), ("proposed", "proposed_yolov8m")]:
            s, c = evaluate.evaluate(RUNS / d / "weights" / "best.pt", n, DATA_YAML, RUNS,
                                     cfg["train"]["imgsz"], cfg["train"]["batch"], "val", dev)
            print(n, {k: round(v, 4) for k, v in s.items()}); print(c.round(4))
