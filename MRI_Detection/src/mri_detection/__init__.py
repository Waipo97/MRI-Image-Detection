"""MRI_Detection: YOLOv8m baseline vs. YOLOv8m + residual CBAM + weighted multi-scale fusion
for brain-tumour detection in MRI."""
import os as _os

# torch 2.6+ loads checkpoints with weights_only=True, which breaks ultralytics 8.3.40 and our custom layers
_os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as _np

# ultralytics 8.3.40 calls np.trapz, which NumPy 2.x renamed to np.trapezoid.
if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid

from .config import ROOT, load_config, get_paths  # noqa: E402,F401
