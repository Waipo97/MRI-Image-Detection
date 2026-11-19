"""Training entry points. Baseline and improved models use exactly the same training arguments."""
from pathlib import Path
from ultralytics import YOLO
from . import modules as cm
from .config import train_args
from .models import save_cfg, make_transfer_cb, pretrained_weights


def train_baseline(cfg, data_yaml, runs_dir, name="baseline_yolov8m", weights=None):
    cm.register(weighted_fusion=False)
    weights = weights or pretrained_weights()
    model = YOLO(weights)
    model.train(name=name, **train_args(cfg, data_yaml, runs_dir))
    return Path(runs_dir) / name / "weights" / "best.pt"


def train_improved(cfg, data_yaml, runs_dir, name="proposed_yolov8m", nc=4,
                   attention=True, weighted=True, weights=None):
    weights = weights or pretrained_weights()
    path, old2new = save_cfg(name, attention=attention, nc=nc)
    cm.register(weighted_fusion=weighted)            # must stay active while the trainer builds the model
    try:
        model = YOLO(path)
        model.add_callback("on_pretrain_routine_end", make_transfer_cb(old2new, weights))
        model.train(name=name, pretrained=False, **train_args(cfg, data_yaml, runs_dir))
    finally:
        cm.register(weighted_fusion=False)
    return Path(runs_dir) / name / "weights" / "best.pt"
