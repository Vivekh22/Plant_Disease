# HMLFF-Net — Hybrid Multi-Level Feature Fusion Network

**Multi-Crop Disease Prediction using Deep Learning** (Tomato · Maize · Chilli)

A complete, research-grade Python project suitable for Q1-journal submission
(*Computers and Electronics in Agriculture*, *Expert Systems with Applications*,
*Knowledge-Based Systems*, *Engineering Applications of Artificial Intelligence*,
*Information Sciences*).

---

## 1. Highlights

| Module | Capability |
|---|---|
| **Preprocessing** | Resize 224×224, CLAHE, Gamma correction, contrast stretching, median + Gaussian denoise, gray-world white balance, color normalization |
| **Segmentation** | YOLOv11 (Ultralytics) with automatic fallback to classical HSV/LAB-Otsu + lightweight U-Net refinement; saves segmented diseased regions |
| **Handcrafted features** | RGB/HSV/LAB stats (mean·std·var·skew·kurt), GLCM (contrast·correlation·energy·homogeneity·entropy·ASM·dissimilarity), LBP histogram, shape (area·perimeter·circularity·aspect·convex-hull·solidity·extent·equiv-diameter·disease %), 256-bin RGB + HSV color histograms |
| **Deep features** | EfficientNetV2 (1280-d) + MobileViT (640-d) via `timm` |
| **Transformer features** | Vision Transformer ViT-B/16 (768-d) + MobileViT (640-d) |
| **Fusion (novel)** | Adaptive Feature Fusion Module — stream-wise channel attention + adaptive scaling + gated aggregation + residual MLP |
| **Selection** | SHAP importance + Mutual Information + Recursive Feature Elimination |
| **Export** | `Feature_Dataset.xlsx` (auto columns) + CSV + SQLite database |
| **Augmentation** | Rotation / flip / brightness / contrast / crop / zoom / CutMix / MixUp / Random Erasing / Elastic Transform |
| **Training** | 5-fold Stratified CV on 7 models: ResNet50, EfficientNetV2, DenseNet121, MobileNetV3, ViT, MobileViT, **HMLFF-Net** |
| **Optimization** | Bayesian optimization (Optuna) over LR / dropout / batch size / weight decay / epochs / optimizer |
| **Loss** | CrossEntropy · Focal · Weighted (class-imbalance aware) |
| **Evaluation** | Accuracy, Precision, Recall, Specificity, F1, ROC/AUC, PR curve, MCC, Cohen's Kappa, Confusion matrix |
| **Visualization** | Train/val accuracy & loss, ROC, PR, confusion matrix, feature importance, PCA / t-SNE / UMAP |
| **Explainability** | GradCAM, GradCAM++, ScoreCAM, LIME, SHAP visualizations |
| **Export artifacts** | Predicted images, heatmaps, segmented images, Excel, CSV, PDF report |

---

## 2. Project structure

```
hmlff_net/
├── __init__.py
├── config.py            # central configuration
├── utils.py             # logging, seeding, dataset discovery
├── preprocessing.py     # classical image enhancement
├── segmentation.py      # YOLOv11 / U-Net / classical segmentation
├── augmentation.py      # CutMix, MixUp, Albumentations, Torch transforms
├── feature_extraction.py# RGB/HSV/LAB/GLCM/LBP/shape/histogram features
├── deep_features.py     # EfficientNetV2 / MobileViT embeddings
├── transformer_features.py # ViT + MobileViT embeddings
├── feature_fusion.py    # Adaptive Feature Fusion Module (torch + numpy)
├── feature_selection.py # SHAP + MI + RFE
├── excel_export.py      # Excel + CSV export
├── database.py          # SQLite persistence
├── dataset.py           # PyTorch Dataset wrapper
├── training.py          # models, losses, 5-fold CV, Bayesian opt
├── testing.py           # inference + confidence + severity
├── evaluation.py        # full metric suite
├── visualization.py     # matplotlib/seaborn figures
├── gradcam.py           # GradCAM / GradCAM++ / ScoreCAM / LIME / SHAP
├── report.py            # PDF report (ReportLab)
├── main.py              # end-to-end CLI orchestrator
├── requirements.txt
└── README.md
```

---

## 3. Dataset layout

Place images under `hmlff_net/Dataset/` (or pass `--dataset <path>`):

```
Dataset/
    Tomato/
        Healthy/
        Disease1/
        Disease2/
        Disease3/
    Maize/
        Healthy/
        Disease1/
        Disease2/
        Disease3/
    Chilli/
        Healthy/
        Disease1/
        Disease2/
        Disease3/
```

Crop/disease label mapping lives in `config.py → CROP_DISEASE_MAP`. Rename
`Disease1/2/3` semantic disease names there to match your dataset.

---

## 4. Installation

**Python 3.12** is recommended.

```bash
# Create a virtual environment
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r hmlff_net/requirements.txt
```

GPU (CUDA) is detected automatically via PyTorch; CPU is used otherwise.

---

## 5. Usage

Run the **complete** pipeline (feature extraction → selection → training →
XAI → PDF report):

```bash
python -m hmlff_net.main --pipeline all
```

Run individual stages:

```bash
python -m hmlff_net.main --pipeline features    # extract features + Excel/CSV/SQLite
python -m hmlff_net.main --pipeline selection    # SHAP/MI/RFE feature selection
python -m hmlff_net.main --pipeline train        # 5-fold CV training of 7 models
python -m hmlff_net.main --pipeline xai          # GradCAM / LIME / SHAP
python -m hmlff_net.main --pipeline report       # PDF report
```

Point at a custom dataset directory:

```bash
python -m hmlff_net.main --pipeline all --dataset /path/to/Dataset
```

All artifacts are written under `hmlff_net/outputs/`:

```
outputs/
├── Feature_Dataset.xlsx
├── Feature_Dataset.csv
├── features.db
├── segmented/      # per-image segmented crops
├── heatmaps/       # GradCAM/LIME/SHAP overlays
├── figures/        # accuracy/loss/ROC/PR/confusion/PCA/tSNE/UMAP
├── models/         # saved checkpoints
└── reports/Final_Report.pdf
```

---

## 6. Hyperparameter optimization

Bayesian (Optuna) search optimizes learning rate, dropout, batch size,
weight decay, epochs and optimizer for any model:

```python
from hmlff_net.training import bayesian_optimize
from hmlff_net.main import run_feature_extraction
df = run_feature_extraction()
# build items then:
# best = bayesian_optimize("HMLFFNet", items, cfg.NUM_CLASSES, n_trials=20)
```

---

## 7. Extending the disease mapping

Edit `config.py`:

```python
CROP_DISEASE_MAP = {
    "Tomato/Disease1": ("Tomato", "Bacterial Spot"),
    ...
}
```

The number of classes and every column name are derived automatically.

---

## 8. Notes

* YOLOv11 weights download automatically on first use (Ultralytics). If
  offline, segmentation falls back to the classical HSV + Otsu + U-Net pipeline.
* `timm` downloads pretrained weights for every backbone on first use.
* Large datasets: feature extraction is single-process but tqdm-instrumented;
  lower `cfg.IMG_SIZE` or run on GPU for speed.
* For the proposed model the fusion module is trainable end-to-end with the
  classifier head; a numpy mirror is used during offline feature extraction.

---

## 9. Citation

If this code helps your research, please cite the accompanying paper
(*HMLFF-Net: Hybrid Multi-Level Feature Fusion Network for Multi-Crop
Disease Prediction using Deep Learning*).

## 10. License

MIT License.