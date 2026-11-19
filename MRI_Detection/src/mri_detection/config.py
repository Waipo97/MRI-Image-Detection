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
