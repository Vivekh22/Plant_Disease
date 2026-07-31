"""
augmentation.py
Rich data augmentation: geometric, photometric, CutMix, MixUp, Random
Erasing, Elastic Transform.  Built on Albumentations + custom torch ops.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger("hmlff_net.augmentation")


class Augmentor:
    """Standard (non-mixing) augmentation via Albumentations."""

    def __init__(self, p: float = 0.5) -> None:
        import albumentations as A

        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05),
            A.CoarseDropout(
                max_holes=8, max_height=24, max_width=24,
                min_holes=1, fill_value=0, p=0.3,
            ),  # Random Erasing analogue
            A.GaussNoise(p=0.2),
        ], p=p)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image=image)["image"]


# ---------------------------------------------------------------------------
# CutMix / MixUp (operate on batches)
# ---------------------------------------------------------------------------
def cutmix(
    images: np.ndarray, labels: np.ndarray, alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """CutMix a batch. Returns (x, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(images))
    bb = _rand_bbox(images.shape[2:], lam)
    x = images.copy()
    x[:, :, bb[0]:bb[2], bb[1]:bb[3]] = images[idx, :, bb[0]:bb[2], bb[1]:bb[3]]
    y_a, y_b = labels, labels[idx]
    return x, y_a, y_b, lam


def mixup(
    images: np.ndarray, labels: np.ndarray, alpha: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """MixUp a batch. Returns (x, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(images))
    x = images * lam + images[idx] * (1 - lam)
    return x, labels, labels[idx], lam


def _rand_bbox(size: Tuple[int, int], lam: float) -> Tuple[int, int, int, int]:
    h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    ch = int(h * cut_rat)
    cw = int(w * cut_rat)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = np.clip(cy - ch // 2, 0, h)
    y2 = np.clip(cy + ch // 2, 0, h)
    x1 = np.clip(cx - cw // 2, 0, w)
    x2 = np.clip(cx + cw // 2, 0, w)
    return y1, x1, y2, x2


# ---------------------------------------------------------------------------
# Torch tensor augmentation wrapper (for training loop)
# ---------------------------------------------------------------------------
class TorchAugmentor:
    """Wraps torchvision transforms for the PyTorch training pipeline."""

    def __init__(self, train: bool = True) -> None:
        import torchvision.transforms as T

        if train:
            self.t = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.ToTensor(),
                T.RandomErasing(p=0.25),
            ])
        else:
            self.t = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

    def __call__(self, img: np.ndarray):
        return self.t(img)