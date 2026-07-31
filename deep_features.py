"""
deep_features.py
Deep CNN feature extraction using timm backbones:
    - EfficientNetV2  -> 1280 features
    - MobileViT       -> 640 features
Features are pooled from the penultimate stage of each model on GPU if
available, otherwise CPU.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.deep_features")

# Lazy torch import; cached after first call.
_torch = None
_nn = None


def _import_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch, _nn = torch, nn
    return _torch, _nn


def cv2_to_tensor(img, size, mean, std, torch):
    """Build a normalized float tensor from a BGR uint8/float image."""
    import cv2
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0
               else np.clip(img * 255, 0, 255).astype(np.uint8))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size[::-1])
    x = rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1)


class TimmFeatureExtractor:
    """Generic timm feature extractor returning a fixed-length embedding."""

    def __init__(self, model_name: str, output_dim: int, device: str | None = None):
        torch, nn = _import_torch()
        import timm

        self.torch = torch
        self.device = device or cfg.DEVICE
        self.output_dim = output_dim
        self.model_name = model_name
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool="avg",
        ).to(self.device).eval()
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.size = tuple(data_cfg["input_size"][-2:])
        self.mean = np.array(data_cfg["mean"], dtype=np.float32)
        self.std = np.array(data_cfg["std"], dtype=np.float32)
        log.info("Loaded %s (dim=%d).", model_name, output_dim)

    def _preprocess(self, img: np.ndarray):
        x = cv2_to_tensor(img, self.size, self.mean, self.std, self.torch)
        return x.unsqueeze(0).to(self.device)

    def extract(self, img: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self._preprocess(img)
            feat = self.model(x).squeeze(0)
            # Ensure exact length
            if feat.shape[0] != self.output_dim:
                feat = self.torch.nn.functional.adaptive_avg_pool1d(
                    feat.unsqueeze(0).unsqueeze(0), self.output_dim
                ).squeeze(0).squeeze(0)
            return feat.detach().cpu().numpy().astype(np.float32)


class DeepFeatureBank:
    """Container that extracts and concatenates CNN deep features."""

    def __init__(self) -> None:
        self.effnet = TimmFeatureExtractor(
            "tf_efficientnetv2_s.in21k_ft_in1k", cfg.NUM_DEEP_CNN
        )

    def extract(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        eff = self.effnet.extract(img)
        return {"efficientnet": eff, "deep_concat": eff}

    @staticmethod
    def names_cnn() -> List[str]:
        return [f"effnet_{i}" for i in range(cfg.NUM_DEEP_CNN)]