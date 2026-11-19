"""Model construction: baseline YOLOv8m and the improved model (residual CBAM + weighted fusion)."""
import yaml
from pathlib import Path
from . import modules as cm
from .config import ROOT

BACKBONE = [
    [-1, 1, "Conv", [64, 3, 2]],      # 0  P1/2
    [-1, 1, "Conv", [128, 3, 2]],     # 1  P2/4
    [-1, 3, "C2f", [128, True]],      # 2
    [-1, 1, "Conv", [256, 3, 2]],     # 3  P3/8
    [-1, 6, "C2f", [256, True]],      # 4
    [-1, 1, "Conv", [512, 3, 2]],     # 5  P4/16
    [-1, 6, "C2f", [512, True]],      # 6
    [-1, 1, "Conv", [1024, 3, 2]],    # 7  P5/32
    [-1, 3, "C2f", [1024, True]],     # 8
    [-1, 1, "SPPF", [1024, 5]],       # 9
]


def build_cfg(attention=True, nc=4):
    """Return (yaml dict, old2new). old2new maps YOLOv8 layer indices to indices in the new model;
    it is used to copy COCO-pretrained YOLOv8m weights into the improved model.
    Channels are for the 'm' scale (width 0.75): P3=192, P4=384, P5/SPPF=576."""
    layers, old2new = list(BACKBONE), {i: i for i in range(10)}

    def add(layer, old=None):
        layers.append(layer)
        n = len(layers) - 1
        if old is not None:
            old2new[old] = n
        return n

    sppf = add([-1, 1, "CBAMAttention", [576]]) if attention else 9
    add([-1, 1, "nn.Upsample", [None, 2, "nearest"]], 10)
    add([[-1, 6], 1, "Concat", [1]], 11)
    c12 = add([-1, 3, "C2f", [512]], 12)
    add([-1, 1, "nn.Upsample", [None, 2, "nearest"]], 13)
    add([[-1, 4], 1, "Concat", [1]], 14)
    p3 = add([-1, 3, "C2f", [256]], 15)
    if attention:
        p3 = add([-1, 1, "CBAMAttention", [192]])
    add([-1, 1, "Conv", [256, 3, 2]], 16)
    add([[-1, c12], 1, "Concat", [1]], 17)
    p4 = add([-1, 3, "C2f", [512]], 18)
    if attention:
        p4 = add([-1, 1, "CBAMAttention", [384]])
    add([-1, 1, "Conv", [512, 3, 2]], 19)
    add([[-1, sppf], 1, "Concat", [1]], 20)
    p5 = add([-1, 3, "C2f", [1024]], 21)
    if attention:
        p5 = add([-1, 1, "CBAMAttention", [576]])
    add([[p3, p4, p5], 1, "Detect", ["nc"]], 22)
    cfg = {"nc": nc, "scale": "m", "scales": {"m": [0.67, 0.75, 768]},
           "backbone": BACKBONE, "head": layers[10:]}
    return cfg, old2new


def save_cfg(name, attention=True, nc=4, out_dir=None):
    import os
    out_dir = Path(out_dir or Path(os.environ.get("MRI_RUNS_DIR") or ROOT / "runs") / "_model_cfg")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, old2new = build_cfg(attention, nc)
    path = out_dir / f"yolov8m_{name}.yaml"      # 'yolov8m' in the name lets ultralytics detect scale 'm'
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(path), old2new


def pretrained_weights(name="yolov8m.pt"):
    """Path to COCO weights in ./weights (downloaded on first use)."""
    from ultralytics.utils.downloads import attempt_download_asset
    import os
    p = Path(os.environ.get("MRI_WEIGHTS_DIR") or ROOT / "weights") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(attempt_download_asset(p))


def build_network(kind="improved", nc=4, attention=True, weighted=True):
    """Instantiate an (untrained) ultralytics DetectionModel for inspection."""
    from ultralytics.nn.tasks import DetectionModel
    if kind == "baseline":
        cm.register(weighted_fusion=False)
        return DetectionModel("yolov8m.yaml", nc=nc, verbose=False)
    path, _ = save_cfg("proposed", attention=attention, nc=nc)
    cm.register(weighted_fusion=weighted)
    try:
        return DetectionModel(path, nc=nc, verbose=False)
    finally:
        cm.register(weighted_fusion=False)


def make_transfer_cb(old2new, weights):
    """Trainer callback: copy every pretrained YOLOv8m tensor whose (remapped) name and shape match
    into the improved model; new layers (CBAM, fusion weights, class head) keep their initial values."""
    from ultralytics import YOLO

    def remap(sd):
        out = {}
        for k, v in sd.items():
            p = k.split(".")
            if p[0] == "model" and p[1].isdigit():
                p[1] = str(old2new.get(int(p[1]), int(p[1])))
            out[".".join(p)] = v
        return out

    def cb(trainer):
        old = remap(YOLO(weights).model.float().state_dict())
        new_sd = trainer.model.state_dict()
        ok = {k: v for k, v in old.items() if k in new_sd and v.shape == new_sd[k].shape}
        trainer.model.load_state_dict(ok, strict=False)
        trainer.ema.ema.load_state_dict(ok, strict=False)
        print(f"[transfer] copied {len(ok)}/{len(old)} pretrained tensors into the improved model "
              f"(model has {len(new_sd)} tensors; the rest are new layers or the new class head)")
    return cb
