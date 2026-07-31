"""
transformer_features.py
Transformer-based feature extraction:
    - Vision Transformer (ViT-B/16) -> 768-d embedding
    - MobileViT                   -> 640-d embedding (timm)
Both backbones run through timm for a single, consistent API.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from . import config as cfg
from .deep_features import TimmFeatureExtractor, torch_inference

log = logging.getLogger("hmlff_net.transformer_features")


class TransformerFeatureBank:
    """Extract ViT and MobileViT embeddings for fusion."""

    def __init__(self) -> None:
        self.vit = TimmFeatureExtractor(
            "vit_base_patch16_224.augreg_in1k", cfg.NUM_DEEP_VIT
        )
        self.mobilevit = TimmFeatureExtractor(
            "mobilevit_s.cvnets_in1k", cfg.NUM_DEEP_MOBILEVIT
        )

    @torch_inference
    def extract(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        vit = self.vit.extract(img)
        mvit = self.mobilevit.extract(img)
        concat = np.concatenate([vit, mvit], axis=0)
        return {
            "vit": vit,
            "mobilevit": mvit,
            "transformer_concat": concat,
        }

    @staticmethod
    def names_transformer() -> List[str]:
        return (
            [f"vit_{i}" for i in range(cfg.NUM_DEEP_VIT)] +
            [f"mobilevit_{i}" for i in range(cfg.NUM_DEEP_MOBILEVIT)]
        )