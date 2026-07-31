"""
segmentation.py
Diseased-region segmentation.

Strategy:
  1. Try YOLOv11 (ultralytics) segmentation -- best quality if a checkpoint
     is available.
  2. Fall back to a classical+U-Net hybrid:
        a. Otsu/HSV mask to isolate non-green (diseased) regions.
        b. Optional lightweight U-Net refinement (torch).
  3. Crop the largest diseased blob, save the segmented image, and return
     the disease fraction for severity scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.segmentation")


@dataclass
class SegmentationResult:
    mask: np.ndarray              # uint8 0/255
    segmented: np.ndarray        # uint8 BGR, background blacked-out
    disease_fraction: float      # area(mask) / area(image)
    crop_box: Optional[Tuple[int, int, int, int]]  # x,y,w,h
    source: str                   # "yolo" | "classical" | "unet"


class YoloSegmenter:
    """Thin wrapper around ultralytics YOLOv11-seg."""

    def __init__(self, weights: str | None = None) -> None:
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights or "yolov8n-seg.pt")
            log.info("YOLO segmenter loaded.")
        except Exception as e:  # pragma: no cover
            log.warning("YOLO unavailable (%s); will use classical/U-Net.", e)
            self.model = None

    def segment(self, img: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        res = self.model.predict(img, verbose=False, conf=0.25)
        if not res or res[0].masks is None:
            return None
        masks = res[0].masks.data.cpu().numpy()  # (N,H,W)
        mask = (masks.sum(0) > 0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        return mask


class ClassicalSegmenter:
    """HSV + LAB + Otsu diseased-region segmentation (no training)."""

    def __init__(self) -> None:
        self.green_lo = np.array([20, 20, 20])
        self.green_hi = np.array([90, 255, 255])

    def segment(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Healthy foliage is green -> the *inverse* is the diseased candidate.
        green = cv2.inRange(hsv, self.green_lo, self.green_hi)
        non_green = cv2.bitwise_not(green)
        #LAB L-channel Otsu for extra robustness on dark lesions.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, otsu = cv2.threshold(
            lab[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        mask = cv2.bitwise_and(non_green, otsu)
        # Morphological cleanup.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, 1)
        # Keep largest blob.
        mask = self._largest_component(mask)
        return mask

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8) * 255


class UNetRefiner:
    """Tiny U-Net for binary mask refinement.

    Used only when a trained checkpoint exists at ``config.MODEL_DIR/unet.pth``.
    Otherwise it is a no-op pass-through.
    """

    def __init__(self) -> None:
        self.net = None
        self.device = cfg.DEVICE
        try:
            import torch
            import torch.nn as nn

            class DoubleConv(nn.Module):
                def __init__(self, i, o):
                    super().__init__()
                    self.d = nn.Sequential(
                        nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                        nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True),
                    )
                def forward(self, x): return self.d(x)

            class UNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.e1 = DoubleConv(3, 16)
                    self.e2 = DoubleConv(16, 32)
                    self.e3 = DoubleConv(32, 64)
                    self.p = nn.MaxPool2d(2)
                    self.b = DoubleConv(64, 128)
                    self.u3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                    self.d3 = DoubleConv(128, 64)
                    self.u2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                    self.d2 = DoubleConv(64, 32)
                    self.u1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
                    self.d1 = DoubleConv(32, 16)
                    self.out = nn.Conv2d(16, 1, 1)

                def forward(self, x):
                    e1 = self.e1(x); e2 = self.e2(self.p(e1)); e3 = self.e3(self.p(e2))
                    b = self.b(self.p(e3))
                    d3 = self.d3(torch.cat([self.u3(b), e3], 1))
                    d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
                    d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
                    return torch.sigmoid(self.out(d1))

            ckpt = cfg.MODEL_DIR / "unet.pth"
            if ckpt.exists():
                self.net = UNet().to(self.device)
                self.net.load_state_dict(torch.load(ckpt, map_location=self.device))
                self.net.eval()
                log.info("U-Net refiner loaded from %s", ckpt)
        except Exception as e:  # pragma: no cover
            log.warning("U-Net refiner unavailable (%s).", e)

    def refine(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.net is None:
            return mask
        import torch
        x = cv2.resize(img, (128, 128)).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.net(x).squeeze().cpu().numpy()
        y = (y > 0.5).astype(np.uint8) * 255
        return cv2.resize(y, (img.shape[1], img.shape[0]))


class SegmentationPipeline:
    """Top-level orchestrator: YOLO -> classical -> U-Net refinement."""

    def __init__(self, use_yolo: bool = True) -> None:
        self.yolo = YoloSegmenter() if use_yolo else None
        self.classical = ClassicalSegmenter()
        self.unet = UNetRefiner()

    def segment(self, img: np.ndarray, save_path: Path | None = None) -> SegmentationResult:
        h, w = img.shape[:2]
        # 1) YOLO
        mask = None
        if self.yolo is not None:
            mask = self.yolo.segment(img)
        source = "yolo" if mask is not None else "classical"
        # 2) Classical fallback
        if mask is None:
            mask = self.classical.segment(img)
        # 3) U-Net refinement
        mask = self.unet.refine(img, mask)
        if source == "classical" and self.unet.net is not None:
            source = "unet"

        # Apply mask
        segmented = cv2.bitwise_and(img, img, mask=mask)
        disease_fraction = float((mask > 0).sum()) / (h * w)

        # Bounding box of largest component
        crop_box = self._bbox(mask)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), segmented)
        return SegmentationResult(
            mask=mask, segmented=segmented,
            disease_fraction=disease_fraction, crop_box=crop_box, source=source,
        )

    @staticmethod
    def _bbox(mask: np.ndarray):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)