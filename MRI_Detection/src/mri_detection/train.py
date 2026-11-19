"""Training entry points. Baseline and improved models use exactly the same training arguments.

Both are resume-safe (important on Colab, where sessions disconnect): if <runs>/<name>/weights/last.pt
exists and the run has no DONE marker, training resumes from it; if DONE exists, training is skipped."""
from pathlib import Path
from ultralytics import YOLO
from . import modules as cm
from .config import train_args
from .models import save_cfg, make_transfer_cb, pretrained_weights


def _state(runs_dir, name):
    run = Path(runs_dir) / name
    best, last, done = run / "weights" / "best.pt", run / "weights" / "last.pt", run / "DONE"
    return best, last, done


def _resume(last):
    """Resume from last.pt. Returns False if that run had in fact already finished."""
    try:
        YOLO(str(last)).train(resume=True)
    except AssertionError as e:
        if "finished" not in str(e):
            raise
        print("[resume] run had already finished:", e)
    return True


def train_baseline(cfg, data_yaml, runs_dir, name="baseline_yolov8m", weights=None):
    best, last, done = _state(runs_dir, name)
    if done.exists() and best.exists():
        print(f"[skip] {name} already finished -> {best}")
        return best
    cm.register(weighted_fusion=False)
    if last.exists():
        print(f"[resume] {name} from {last}")
        _resume(last)
    else:
        model = YOLO(weights or pretrained_weights())
        model.train(name=name, **train_args(cfg, data_yaml, runs_dir))
    done.write_text("ok")
    return best


def train_improved(cfg, data_yaml, runs_dir, name="proposed_yolov8m", nc=4,
                   attention=True, weighted=True, weights=None):
    best, last, done = _state(runs_dir, name)
    if done.exists() and best.exists():
        print(f"[skip] {name} already finished -> {best}")
        return best
    cm.register(weighted_fusion=weighted)            # must stay active while the trainer builds the model
    try:
        if last.exists():
            print(f"[resume] {name} from {last}")
            _resume(last)                            # no transfer callback: it would overwrite trained weights
        else:
            weights = weights or pretrained_weights()
            path, old2new = save_cfg(name, attention=attention, nc=nc)
            model = YOLO(path)
            model.add_callback("on_pretrain_routine_end", make_transfer_cb(old2new, weights))
            model.train(name=name, pretrained=False, **train_args(cfg, data_yaml, runs_dir))
    finally:
        cm.register(weighted_fusion=False)
    done.write_text("ok")
    return best
