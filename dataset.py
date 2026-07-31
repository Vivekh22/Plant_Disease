"""
dataset.py
PyTorch Dataset wrappers used by the training loop.  Kept separate from
main.py for clarity and reusability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.dataset")

_torch = None


def _torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class CropDiseaseDataset:
    """In-memory image dataset.

    items: list of dicts with keys
        {"path": Path, "image": np.ndarray (preprocessed),
         "label_idx": int, "hand": Optional[np.ndarray]}
    """

    def __init__(self, items: List[dict], transform: Optional[Callable] = None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = item["image"]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = _default_tensor(img)
        out = {"image": img, "label": item["label_idx"]}
        if item.get("hand") is not None:
            out["hand"] = _torch().from_numpy(
                item["hand"].astype(np.float32))
        return out


def _default_tensor(img: np.ndarray):
    import torchvision.transforms as T
    torch = _torch()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0
               else np.clip(img * 255, 0, 255).astype(np.uint8))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    x = rgb.astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x).permute(2, 0, 1)