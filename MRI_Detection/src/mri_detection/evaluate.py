import pandas as pd
from ultralytics import YOLO
from . import modules as cm


def evaluate(weights, name, data_yaml, runs_dir, imgsz=640, batch=16, split="val", device=None):
    """Validate a checkpoint. Returns (overall dict, per-class DataFrame)."""
    cm.register(weighted_fusion=False)           # custom layers are unpickled from the checkpoint
    m = YOLO(str(weights))
    # size/compute are measured on the unfused model, before validation fuses Conv+BN
    params = sum(p.numel() for p in m.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(m.model, imgsz))
    except Exception:
        gflops = float("nan")
    r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, plots=True, device=device,
              project=str(runs_dir), name=f"{name}_{split}", exist_ok=True, verbose=False)
    P, R = float(r.box.mp), float(r.box.mr)
    summary = {"Precision": P, "Recall": R, "F1": 2 * P * R / (P + R + 1e-9),
               "mAP@50": float(r.box.map50), "mAP@50-95": float(r.box.map),
               "Params (M)": params / 1e6, "GFLOPs": gflops, "Inference (ms/img)": r.speed["inference"]}
    rows = []
    for i, c in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap = r.box.class_result(i)
        rows.append({"Class": r.names[int(c)], "P": p, "R": rc, "F1": 2 * p * rc / (p + rc + 1e-9),
                     "mAP@50": ap50, "mAP@50-95": ap})
    return summary, pd.DataFrame(rows)


def load_curve(runs_dir, name):
    d = pd.read_csv(f"{runs_dir}/{name}/results.csv")
    d.columns = d.columns.str.strip()
    return d
