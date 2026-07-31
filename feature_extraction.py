"""
feature_extraction.py
Handcrafted feature extraction covering RGB, HSV, LAB, GLCM, LBP, shape
and color-histogram descriptors.  All features are concatenated into a
single float32 vector with a stable, deterministic ordering so that
column names can be generated automatically downstream.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from . import config as cfg

log = logging.getLogger("hmlff_net.feature_extraction")


# ---------------------------------------------------------------------------
# Feature-name registry (column names are auto-generated from this)
# ---------------------------------------------------------------------------
RGB_NAMES = [f"rgb_mean_{c}" for c in "rgb"] + \
            [f"rgb_std_{c}" for c in "rgb"] + \
            [f"rgb_var_{c}" for c in "rgb"] + \
            [f"rgb_skew_{c}" for c in "rgb"] + \
            [f"rgb_kurt_{c}" for c in "rgb"]
HSV_NAMES = [f"hsv_mean_{c}" for c in "hsv"] + [f"hsv_std_{c}" for c in "hsv"]
LAB_NAMES = [f"lab_mean_{c}" for c in "lab"]
GLCM_NAMES = [f"glcm_{p}" for p in
              ["contrast", "correlation", "energy", "homogeneity",
               "entropy", "asm", "dissimilarity"]]
LBP_NAMES = [f"lbp_bin_{i:03d}" for i in range(26)]
SHAPE_NAMES = ["area", "perimeter", "circularity", "aspect_ratio",
               "convex_hull", "convex_area", "solidity", "extent",
               "bbox", "equivalent_diameter", "disease_pct"]
HIST_NAMES = [f"rgb_hist_{c}_bin_{i:03d}" for c in "bgr" for i in range(256)] + \
             [f"hsv_hist_{c}_bin_{i:03d}" for c in "hsv" for i in range(256)]

FEATURE_NAMES: List[str] = (
    RGB_NAMES + HSV_NAMES + LAB_NAMES + GLCM_NAMES +
    LBP_NAMES + SHAPE_NAMES + HIST_NAMES
)


def count_handcrafted() -> int:
    return len(FEATURE_NAMES)


cfg.NUM_HANDCRAFTED = count_handcrafted()


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def _stats(channel: np.ndarray) -> Tuple[float, float, float, float, float]:
    flat = channel.astype(np.float64).ravel()
    mean = flat.mean()
    std = flat.std()
    var = flat.var()
    # skewness / kurtosis (Fisher)
    n = flat.size
    if std < 1e-9:
        return float(mean), float(std), float(var), 0.0, 0.0
    z = (flat - mean) / std
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)
    return float(mean), float(std), float(var), skew, kurt


# ---------------------------------------------------------------------------
# Extractors (each returns a list aligned to its *_NAMES list)
# ---------------------------------------------------------------------------
def rgb_features(img: np.ndarray) -> List[float]:
    out = []
    for i in range(3):  # B, G, R
        m, s, v, sk, ku = _stats(img[:, :, i])
        out += [m, s, v, sk, ku]
    return out


def hsv_features(img: np.ndarray) -> List[float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for i in range(3):
        flat = hsv[:, :, i].astype(np.float64).ravel()
        out += [float(flat.mean()), float(flat.std())]
    return out


def lab_features(img: np.ndarray) -> List[float]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return [float(lab[:, :, i].mean()) for i in range(3)]


def glcm_features(gray: np.ndarray) -> List[float]:
    g = (gray // 4).astype(np.uint8)  # quantize to 64 levels
    glcm = graycomatrix(g, distances=[1, 3, 5],
                       angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                       levels=64, symmetric=True, normed=True)
    props = ["contrast", "correlation", "homogeneity", "energy", "ASM"]
    vals = []
    for p in props:
        try:
            vals.append(float(graycoprops(glcm, p).mean()))
        except Exception:
            vals.append(0.0)
    # entropy
    glcm_norm = glcm / (glcm.sum() + 1e-9)
    entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-9))
    vals.append(float(entropy.mean()))
    # dissimilarity
    vals.append(float(graycoprops(glcm, "dissimilarity").mean()))
    # ASM already as "energy" alias -> also expose as asm
    asm = float(graycoprops(glcm, "ASM").mean())
    # reorder to match GLCM_NAMES:
    # contrast, correlation, energy, homogeneity, entropy, asm, dissimilarity
    contrast, correlation, energy, homogeneity, _ = vals[:5]
    entropy_v = vals[5]
    dissimilarity = vals[6]
    return [contrast, correlation, energy, homogeneity, entropy_v, asm, dissimilarity]


def lbp_features(gray: np.ndarray, P: int = 8, R: float = 1.0,
                 bins: int = 26) -> List[float]:
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, P + 2),
                           density=True)
    return hist.tolist()


def shape_features(mask: np.ndarray, disease_fraction: float) -> List[float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0] * len(SHAPE_NAMES)
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    hull = cv2.convexHull(c)
    convex_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(c)
    bbox = float(w * h)
    equiv_diam = float(2 * np.sqrt(area / np.pi))
    circularity = float((4 * np.pi * area) / (perim ** 2)) if perim > 0 else 0.0
    aspect_ratio = float(w / h) if h > 0 else 0.0
    solidity = float(area / convex_area) if convex_area > 0 else 0.0
    extent = float(area / bbox) if bbox > 0 else 0.0
    return [
        area, perim, circularity, aspect_ratio,
        float(cv2.contourArea(hull)), convex_area, solidity, extent,
        bbox, equiv_diam, disease_fraction,
    ]


def color_histograms(img: np.ndarray) -> List[float]:
    h_rgb, h_hsv = [], []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        h_rgb.extend(hist.tolist())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        h_hsv.extend(hist.tolist())
    return h_rgb + h_hsv


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------
class HandcraftedExtractor:
    """Compute and concatenate every handcrafted descriptor."""

    def __init__(self) -> None:
        log.info("Handcrafted feature vector length = %d", len(FEATURE_NAMES))

    def extract(self, img_bgr: np.ndarray, mask: np.ndarray | None = None,
                disease_fraction: float = 0.0) -> Dict[str, List[float]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        m = mask if mask is not None else np.full(gray.shape, 255, np.uint8)
        rgb = rgb_features(img_bgr)
        hsv = hsv_features(img_bgr)
        lab = lab_features(img_bgr)
        glcm = glcm_features(gray)
        lbp = lbp_features(gray)
        shape = shape_features(m, disease_fraction)
        hist = color_histograms(img_bgr)
        full = rgb + hsv + lab + glcm + lbp + shape + hist
        assert len(full) == len(FEATURE_NAMES), \
            f"feature length mismatch {len(full)} vs {len(FEATURE_NAMES)}"
        return {
            "rgb": rgb, "hsv": hsv, "lab": lab, "glcm": glcm,
            "lbp": lbp, "shape": shape, "hist": hist, "full": full,
            "names": list(FEATURE_NAMES),
        }