# ============================================================================
# config.py
# Central configuration for the HMLFF-Net project.
# Author: HMLFF-Net Research Team
# License: MIT
# ============================================================================

"""Global configuration constants and hyperparameters for HMLFF-Net.

Edit the values in this file to adapt the pipeline to your environment.
All paths are resolved relative to ``PROJECT_ROOT`` so the project is
portable across machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
SEGMENTED_DIR: Path = OUTPUT_DIR / "segmented"
HEATMAP_DIR: Path = OUTPUT_DIR / "heatmaps"
PREDICTED_DIR: Path = OUTPUT_DIR / "predicted"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
MODEL_DIR: Path = OUTPUT_DIR / "models"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"

EXCEL_PATH: Path = OUTPUT_DIR / "Feature_Dataset.xlsx"
CSV_PATH: Path = OUTPUT_DIR / "Feature_Dataset.csv"
SQLITE_PATH: Path = OUTPUT_DIR / "features.db"
PDF_REPORT_PATH: Path = REPORT_DIR / "Final_Report.pdf"


def make_dirs() -> None:
    """Create every output directory required by the pipeline."""
    for d in (
        OUTPUT_DIR, SEGMENTED_DIR, HEATMAP_DIR, PREDICTED_DIR,
        REPORT_DIR, MODEL_DIR, FIGURE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Crops and diseases
# ---------------------------------------------------------------------------
CROPS: List[str] = ["Tomato", "Maize", "Chilli"]

# Folder name -> (crop, disease) mapping. Adjust disease names to your dataset.
CROP_DISEASE_MAP: Dict[str, Tuple[str, str]] = {
    # Tomato
    "Tomato/Healthy": ("Tomato", "Healthy"),
    "Tomato/Disease1": ("Tomato", "Bacterial Spot"),
    "Tomato/Disease2": ("Tomato", "Early Blight"),
    "Tomato/Disease3": ("Tomato", "Late Blight"),
    # Maize
    "Maize/Healthy": ("Maize", "Healthy"),
    "Maize/Disease1": ("Maize", "Common Rust"),
    "Maize/Disease2": ("Maize", "Gray Leaf Spot"),
    "Maize/Disease3": ("Maize", "Northern Leaf Blight"),
    # Chilli
    "Chilli/Healthy": ("Chilli", "Healthy"),
    "Chilli/Disease1": ("Chilli", "Leaf Curl"),
    "Chilli/Disease2": ("Chilli", "Bacterial Spot"),
    "Chilli/Disease3": ("Chilli", "Powdery Mildew"),
}


def build_class_index() -> Dict[str, int]:
    """Return a deterministic label -> index mapping."""
    labels = sorted({f"{c}_{d}" for c, d in CROP_DISEASE_MAP.values()})
    return {label: i for i, label in enumerate(labels)}


CLASS_INDEX: Dict[str, int] = build_class_index()
NUM_CLASSES: int = len(CLASS_INDEX)
INDEX_CLASS: Dict[int, str] = {v: k for k, v in CLASS_INDEX.items()}


# ---------------------------------------------------------------------------
# Image / model parameters
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (224, 224)
NUM_HANDCRAFTED: int = 0  # filled at runtime by feature_extraction.count_handcrafted()
NUM_DEEP_CNN: int = 1280          # EfficientNetV2
NUM_DEEP_MOBILEVIT: int = 640     # MobileViT
NUM_DEEP_VIT: int = 768           # ViT base patch16
FUSION_DIM: int = 1024


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def select_device() -> str:
    """Return ``'cuda'`` if a CUDA GPU is available, else ``'cpu'``."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE: str = select_device()
SEED: int = 42


# ---------------------------------------------------------------------------
# Training defaults (overridden by Bayesian optimization)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    n_splits: int = 5
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    dropout: float = 0.3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"            # adam | adamw | sgd
    loss: str = "focal"                # cross_entropy | focal | weighted
    label_smoothing: float = 0.1
    patience: int = 7                  # early stopping
    num_workers: int = 2
    mixed_precision: bool = True


TRAIN_CFG = TrainConfig()


# ---------------------------------------------------------------------------
# Models to train in the comparative study
# ---------------------------------------------------------------------------
MODEL_NAMES: List[str] = [
    "ResNet50",
    "EfficientNetV2",
    "DenseNet121",
    "MobileNetV3",
    "VisionTransformer",
    "MobileViT",
    "HMLFFNet",
]


# Severity thresholds (fraction of diseased segmented area)
SEVERITY_LEVELS: List[Tuple[float, float, str]] = [
    (0.00, 0.05, "None"),
    (0.05, 0.15, "Low"),
    (0.15, 0.35, "Moderate"),
    (0.35, 1.01, "Severe"),
]


def severity_label(disease_fraction: float) -> str:
    for lo, hi, name in SEVERITY_LEVELS:
        if lo <= disease_fraction < hi:
            return name
    return "None"


# Re-export for convenience
__all__ = [
    name for name in dir() if not name.startswith("_")
]