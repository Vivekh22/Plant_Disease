"""
preprocessing.py
Classical image preprocessing: resize, CLAHE, gamma correction, contrast
stretching, median/Gaussian denoise, white balance, color normalization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.preprocessing")


@dataclass
class PreprocessConfig:
    size: Tuple[int, int] = cfg.IMG_SIZE
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    gamma: float = 1.0          # 1.0 disables
    median_ksize: int = 3
    gaussian_ksize: int = 3
    white_balance: bool = True
    color_normalize: bool = True


class ImagePreprocessor:
    """Deterministic, GPU-free classical preprocessing pipeline."""

    def __init__(self, params: PreprocessConfig | None = None) -> None:
        self.p = params or PreprocessConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the full classical pipeline and return a float32 BGR image."""
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to preprocessor.")
        img = image
        if img.dtype != np.uint8:
            img = self._to_uint8(img)
        img = cv2.resize(img, self.p.size, interpolation=cv2.INTER_AREA)
        img = self.denoise(img)
        if self.p.white_balance:
            img = self.white_balance_gray_world(img)
        img = self.contrast_stretch(img)
        if abs(self.p.gamma - 1.0) > 1e-3:
            img = self.gamma_correction(img, self.p.gamma)
        img = self.clahe(img)
        if self.p.color_normalize:
            img = self.color_normalize(img)
        return img.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Individual ops
    # ------------------------------------------------------------------
    def clahe(self, img: np.ndarray) -> np.ndarray:
        """Contrast-Limited Adaptive Histogram Equalization on L channel."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.p.clahe_clip, tileGridSize=self.p.clahe_grid
        )
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 1e-3)
        table = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)]
        ).astype(np.uint8)
        return cv2.LUT(img, table)

    def contrast_stretch(self, img: np.ndarray) -> np.ndarray:
        """Percentile-based contrast stretching per channel."""
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            lo, hi = np.percentile(ch, [2, 98])
            if hi - lo < 1e-3:
                out[:, :, c] = ch
            else:
                out[:, :, c] = np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255)
        return out.astype(np.uint8)

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Median filter followed by Gaussian smoothing."""
        img = cv2.medianBlur(img, self.p.median_ksize)
        if self.p.gaussian_ksize and self.p.gaussian_ksize > 1:
            k = self.p.gaussian_ksize | 1  # ensure odd
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def white_balance_gray_world(self, img: np.ndarray) -> np.ndarray:
        """Gray-world assumption white balance."""
        b, g, r = cv2.split(img.astype(np.float32))
        avg = (b.mean() + g.mean() + r.mean()) / 3.0
        scale = lambda ch: ch * (avg / (ch.mean() + 1e-6))
        return np.clip(cv2.merge(
            (scale(b), scale(g), scale(r))
        ), 0, 255).astype(np.uint8)

    def color_normalize(self, img: np.ndarray) -> np.ndarray:
        """Per-channel zero-mean / unit-variance normalization (uint8 output)."""
        f = img.astype(np.float32)
        f = (f - f.mean(axis=(0, 1))) / (f.std(axis=(0, 1)) + 1e-6)
        return np.clip(f * 128 + 128, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.max() <= 1.0 \
                else np.clip(img, 0, 255).astype(np.uint8)
        return img