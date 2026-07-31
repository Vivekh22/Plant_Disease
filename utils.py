"""
utils.py
Shared utilities: logging, seeding, device helpers, IO, progress bars.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_CONFIGURED = False


def get_logger(name: str = "hmlff_net", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to both stdout and a log file."""
    global _LOG_CONFIGURED
    logger = logging.getLogger(name)
    if _LOG_CONFIGURED and logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        from . import config as cfg
        cfg.make_dirs()
        fh = logging.FileHandler(cfg.OUTPUT_DIR / "run.log", mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    _LOG_CONFIGURED = True
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch / TF RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def discover_images(dataset_dir: Path) -> List[dict]:
    """Walk the Dataset tree and return one record per image.

    Each record is::

        {"path": Path, "rel": "Tomato/Disease1/img.jpg",
         "crop": "Tomato", "disease": "Bacterial Spot",
         "label": "Tomato_Bacterial Spot", "name": "img.jpg"}
    """
    from . import config as cfg

    records: List[dict] = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    if not dataset_dir.exists():
        return records
    for crop_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for sub in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            rel = f"{crop_dir.name}/{sub.name}"
            crop, disease = cfg.CROP_DISEASE_MAP.get(rel, (crop_dir.name, sub.name))
            label = f"{crop}_{disease}"
            for img_path in sorted(sub.glob("*")):
                if img_path.suffix.lower() in exts:
                    records.append({
                        "path": img_path,
                        "rel": rel,
                        "crop": crop,
                        "disease": disease,
                        "label": label,
                        "name": img_path.name,
                    })
    return records


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
def to_iterable(x) -> Iterable:
    if isinstance(x, (list, tuple, set)):
        return x
    return [x]


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0.0, 0) else default