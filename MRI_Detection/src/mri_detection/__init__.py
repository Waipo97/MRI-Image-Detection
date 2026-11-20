"""MRI_Detection: YOLOv8m baseline vs. YOLOv8m + residual CBAM + weighted multi-scale fusion
for brain-tumour detection in MRI."""
import numpy as _np

# ultralytics 8.3.40 calls np.trapz, which NumPy 2.x renamed to np.trapezoid.
if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid

from .config import ROOT, load_config, runs_dir  # noqa: E402,F401
