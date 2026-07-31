"""
feature_fusion.py
HMLFF-Net's core contribution: an **Adaptive Feature Fusion Module** that
combines handcrafted + CNN + transformer feature streams using:

  * Stream-wise channel attention to learn per-feature weights.
  * Residual connections preserving each stream's information.
  * Adaptive feature scaling (learned scale + bias per stream).
  * Gated aggregation producing the final hybrid feature vector.

Implemented in PyTorch so it is differentiable end-to-end during training,
with a numpy mirror for offline feature extraction without gradients.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.feature_fusion")

_torch = None
_nn = None


def _import_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch, _nn = torch, nn
    return _torch, _nn


# ---------------------------------------------------------------------------
# Torch module (lazy build so the module imports even without torch)
# ---------------------------------------------------------------------------
def build_fusion_module(hand_dim, cnn_dim, vit_dim, out_dim=cfg.FUSION_DIM):
    """Construct and return the PyTorch AdaptiveFusion module."""
    torch, nn = _import_torch()

    class StreamAttention(nn.Module):
        def __init__(self, dim: int, reduction: int = 8):
            super().__init__()
            hidden = max(dim // reduction, 8)
            self.fc = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, dim), nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(x)

    class AdaptiveFusion(nn.Module):
        def __init__(self, hand_dim, cnn_dim, vit_dim, out_dim):
            super().__init__()
            self.out_dim = out_dim
            self.proj_hand = nn.Linear(hand_dim, out_dim)
            self.proj_cnn = nn.Linear(cnn_dim, out_dim)
            self.proj_vit = nn.Linear(vit_dim, out_dim)
            self.scale_hand = nn.Parameter(torch.ones(out_dim))
            self.bias_hand = nn.Parameter(torch.zeros(out_dim))
            self.scale_cnn = nn.Parameter(torch.ones(out_dim))
            self.bias_cnn = nn.Parameter(torch.zeros(out_dim))
            self.scale_vit = nn.Parameter(torch.ones(out_dim))
            self.bias_vit = nn.Parameter(torch.zeros(out_dim))
            self.attn_hand = StreamAttention(out_dim)
            self.attn_cnn = StreamAttention(out_dim)
            self.attn_vit = StreamAttention(out_dim)
            self.gate = nn.Sequential(
                nn.Linear(out_dim * 3, 3), nn.Softmax(dim=-1),
            )
            self.fuse = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim), nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(out_dim, out_dim),
            )

        def forward(self, hand, cnn, vit):
            h = self.scale_hand * self.proj_hand(hand) + self.bias_hand
            c = self.scale_cnn * self.proj_cnn(cnn) + self.bias_cnn
            v = self.scale_vit * self.proj_vit(vit) + self.bias_vit
            h = h + self.attn_hand(h)
            c = c + self.attn_cnn(c)
            v = v + self.attn_vit(v)
            stacked = torch.cat([h, c, v], dim=-1)
            g = self.gate(stacked)
            fused = g[:, 0:1] * h + g[:, 1:2] * c + g[:, 2:3] * v
            out = fused + self.fuse(fused)
            return out, g

    return AdaptiveFusion(hand_dim, cnn_dim, vit_dim, out_dim)


# ---------------------------------------------------------------------------
# Numpy mirror (offline feature extraction, no gradients)
# ---------------------------------------------------------------------------
class NumpyAdaptiveFusion:
    def __init__(self, hand_dim, cnn_dim, vit_dim,
                 out_dim: int = cfg.FUSION_DIM, seed: int = cfg.SEED):
        rng = np.random.default_rng(seed)
        self.out_dim = out_dim
        self.params: Dict[str, np.ndarray] = {}
        for name, in_dim in [("hand", hand_dim), ("cnn", cnn_dim), ("vit", vit_dim)]:
            self.params[f"proj_{name}_w"] = rng.normal(0, 0.02, (in_dim, out_dim)).astype(np.float32)
            self.params[f"proj_{name}_b"] = np.zeros(out_dim, np.float32)
            self.params[f"scale_{name}"] = np.ones(out_dim, np.float32)
            self.params[f"bias_{name}"] = np.zeros(out_dim, np.float32)
            self.params[f"attn_{name}_w1"] = rng.normal(0, 0.02, (out_dim, max(out_dim // 8, 8))).astype(np.float32)
            self.params[f"attn_{name}_b1"] = np.zeros(max(out_dim // 8, 8), np.float32)
            self.params[f"attn_{name}_w2"] = rng.normal(0, 0.02, (max(out_dim // 8, 8), out_dim)).astype(np.float32)
            self.params[f"attn_{name}_b2"] = np.zeros(out_dim, np.float32)
        self.params["gate_w"] = rng.normal(0, 0.02, (out_dim * 3, 3)).astype(np.float32)
        self.params["gate_b"] = np.zeros(3, np.float32)

    @staticmethod
    def _linear(x, w, b):
        return x @ w + b

    def _attn(self, x, prefix):
        p = self.params
        h = np.maximum(0, self._linear(x, p[f"{prefix}_w1"], p[f"{prefix}_b1"]))
        s = 1.0 / (1.0 + np.exp(-(self._linear(h, p[f"{prefix}_w2"], p[f"{prefix}_b2"]))))
        return x * s

    def fuse(self, hand, cnn, vit):
        p = self.params
        h = p["scale_hand"] * self._linear(hand, p["proj_hand_w"], p["proj_hand_b"]) + p["bias_hand"]
        c = p["scale_cnn"] * self._linear(cnn, p["proj_cnn_w"], p["proj_cnn_b"]) + p["bias_cnn"]
        v = p["scale_vit"] * self._linear(vit, p["proj_vit_w"], p["proj_vit_b"]) + p["bias_vit"]
        h = h + self._attn(h, "attn_hand")
        c = c + self._attn(c, "attn_cnn")
        v = v + self._attn(v, "attn_vit")
        stacked = np.concatenate([h, c, v], axis=-1)
        logits = stacked @ p["gate_w"] + p["gate_b"]
        g = np.exp(logits - logits.max(axis=-1, keepdims=True))
        g = g / g.sum(axis=-1, keepdims=True)
        fused = g[..., 0:1] * h + g[..., 1:2] * c + g[..., 2:3] * v
        fused = fused + _layernorm(fused)  # residual identity (no trained MLP here)
        return fused, g

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        self.params.update(sd)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return dict(self.params)


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def fusion_feature_names(out_dim: int = cfg.FUSION_DIM) -> List[str]:
    return [f"fused_{i}" for i in range(out_dim)]