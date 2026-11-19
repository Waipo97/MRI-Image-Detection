import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]          # MRI_Detection/


def load_config(path="configs/train.yaml", quick=False):
    """Load the shared config. quick=True applies the small 'quick' overrides (CPU sanity run)."""
    cfg = yaml.safe_load(open(ROOT / path))
    if quick:
        cfg["train"].update(cfg["quick"])
    cfg["quick_mode"] = quick
    return cfg


def get_paths(cfg):
    """Folders used by the pipeline. By default everything stays inside the repo.
    On Colab, MRI_WORK_DIR points to local disk (fast) and MRI_RUNS_DIR to Drive (kept after disconnect)."""
    d = cfg["dataset"]
    work = os.environ.get("MRI_WORK_DIR")
    runs = os.environ.get("MRI_RUNS_DIR")
    base = Path(work) if work else ROOT
    return dict(
        raw_zip=Path(os.environ.get("MRI_RAW_ZIP") or base / d["raw_zip"]),
        raw_dir=base / d["raw_dir"],
        yolo_dir=base / d["yolo_dir"],
        data_yaml=base / d["yolo_dir"] / "data.yaml",
        runs=Path(runs) if runs else ROOT / "runs",
        weights=Path(os.environ.get("MRI_WEIGHTS_DIR") or ROOT / "weights"),
        manifests=(Path(runs) / "manifests") if runs else ROOT / "data",
    )


def resolve_device(device="auto"):
    if device != "auto":
        return device
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def train_args(cfg, data_yaml, project, **extra):
    """Ultralytics .train() kwargs, identical for baseline and improved model."""
    t = dict(cfg["train"])
    t.pop("device", None)
    args = dict(t, data=str(data_yaml), seed=cfg["seed"], device=resolve_device(cfg["train"]["device"]),
                project=str(project), exist_ok=True, plots=True, val=True, save=True)
    args["amp"] = args["device"] != "cpu"
    args.update(extra)
    return args
