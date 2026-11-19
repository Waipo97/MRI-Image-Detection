# MRI_Detection

Brain-tumour detection in MRI with **YOLOv8m** and an improved variant (**residual CBAM attention + learnable weighted multi-scale fusion**). Built for VS Code with Python and Jupyter notebooks, run step by step.

Dataset: Kaggle *Labeled MRI Brain Tumor Dataset* (`archive.zip`; classes: pituitary, meningioma, glioma, notumor).


## Run on Google Colab (recommended for the full experiment)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Waipo97/MRI-Image-Detection/blob/colab-ready/MRI_Detection/MRI_Detection_Colab.ipynb)

1. Put `archive.zip` in Google Drive at `MyDrive/MRI_Detection/archive.zip`.
2. Open the notebook above, choose `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.
3. Everything (checkpoints, tables, plots, split manifests) is written to `MyDrive/MRI_Detection/runs/`.

The notebook installs the pinned `ultralytics`, builds the cleaned YOLO dataset on Colab's fast local disk, trains the baseline and the improved model with the same `configs/train.yaml`, and evaluates both on val and test.
**Disconnects are safe:** re-run all cells; a finished run is skipped and an interrupted one resumes from `last.pt`.

The same pipeline from a terminal: `python scripts/run.py all` (or `prepare`, `baseline`, `improved`, `evaluate`; add `--quick` for a sanity run).
Paths can be redirected with `MRI_WORK_DIR`, `MRI_RUNS_DIR`, `MRI_RAW_ZIP`, `MRI_WEIGHTS_DIR` (this is how the Colab notebook keeps data on local disk and results on Drive).

## Quick start (VS Code)

1. Open the `MRI_Detection` folder in VS Code (install the recommended Python + Jupyter extensions).
2. Create an environment and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt    # on a GPU machine install the CUDA build of torch first
   ```
3. Copy `archive.zip` into `data/raw/`.
4. Pick the `.venv` interpreter as the notebook kernel and run the notebooks in order.

## Steps

| Notebook | What it does | GPU? |
|---|---|---|
| `01_data_overview` | unzip, audit labels, class balance, box sizes, sample images | no |
| `02_dataset_split` | exclude bad labels, stratified 70/20/10 split, write `data/yolo/` + `data.yaml` | no |
| `03_baseline_yolov8m` | model size, shared training config, train the baseline | yes |
| `04_improved_architecture` | build the improved model, params/GFLOPs, sanity checks, weight transfer check | no |
| `05_train_improved` | train the improved model, check learned gates | yes |
| `06_evaluation` | P / R / F1 / mAP, per-class, curves, confusion matrices, qualitative examples | yes |
| `07_ablation_optional` | attention only vs fusion only vs both | yes |

Each training notebook has `QUICK_TEST = True` (3 epochs, 320 px, 25% of train) to confirm the pipeline runs on a CPU. The real experiment uses `QUICK_TEST = False` on a GPU (up to 150 epochs, early stopping patience 40, 640 px).

## Layout

```
configs/train.yaml        one config shared by baseline and improved model
src/mri_detection/        data.py  modules.py  models.py  train.py  evaluate.py  viz.py  config.py
notebooks/                the seven steps above
scripts/run.py            same pipeline from the command line (prepare | baseline | improved | evaluate | all)
MRI_Detection_Colab.ipynb one-click Colab version (Drive for data + results, resumable)
data/raw/                 put archive.zip here          data/yolo/  generated
runs/                     training outputs              weights/    COCO yolov8m.pt (auto-download)
```

## What was found in the dataset (and how it is handled)

- **No class names are shipped**; ids come from the labels: 0 pituitary, 1 meningioma, 2 glioma, 3 notumor (inferred per folder, then checked to be one-to-one).
- **Excluded images (3):** `Tr-gl_0250`, `Tr-gl_0256` (empty label files) and `Tr-no_0559` (a glioma-class box inside the `notumor` folder). Listed in `data/excluded_images.csv`.
- **Stray files ignored:** `classes.txt` (x2) and `gg (9).txt`, which have no image.
- **55 groups of byte-identical images** (129 images): kept, but always in the same split, so they cannot leak from train into val/test.
- `notumor` images carry one large box, so "no tumour" is a fourth detection class.
- Limitation: the split is per image and the dataset has no patient ids, so slices of one patient can appear in different splits.

## Fair comparison

Both models use the same split, seed, image size, optimiser, loss gains and **explicit** geometric augmentation (rotation +/-7 deg, translate 3%, scale 8%, horizontal flip; no mosaic, mixup, copy-paste or colour changes). Your two earlier notebooks trained the baseline with Ultralytics defaults (100 epochs) and the improved model with the explicit setup (150 epochs), which is not a controlled comparison; here both use `configs/train.yaml`.

## Improved model

- CBAM (Woo et al., ECCV 2018): channel attention, then spatial attention (7x7), reduction 16. Inserted after SPPF and after the P3/P4/P5 neck outputs.
- Deviations from the paper: SiLU in the MLP, and a learnable residual gate `y = x + gamma * (CBAM(x) - x)` with `gamma = 0` at the start.
- Every neck `Concat` becomes a BiFPN-style normalised weighted concat (weights start at 1 = plain concat).
- COCO weights are copied into every tensor that still fits (469 of 475), so only the new layers and the class head start from scratch.

## Notes

- `ultralytics` is pinned to 8.3.40 because the custom layers hook into its model parser. Trained improved checkpoints need `src/` importable (the notebooks and `scripts/run.py` do this) to be loaded later.
- Research prototype, not a clinical tool.

## Reference

S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. *CBAM: Convolutional Block Attention Module.* ECCV 2018.
