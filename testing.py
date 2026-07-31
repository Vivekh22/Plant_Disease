"""
testing.py
Inference: run a trained model on a test set, produce per-image
predictions, confidence scores, probabilities and severity estimates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.testing")


def predict_dataset(model, loader, device: str | None = None,
                    is_hmlff: bool = False) -> Dict[str, np.ndarray]:
    """Return arrays: preds, probs, confidence, labels, severities."""
    torch, _ = _import_torch()
    device = device or cfg.DEVICE
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            hand = batch.get("hand")
            if is_hmlff and hand is not None:
                out, _ = model(imgs, hand.to(device))
            else:
                out = model(imgs)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    return {
        "preds": preds,
        "probs": probs,
        "confidence": confidence,
        "labels": labels,
        "severities": np.array([_severity_from_conf(c) for c in confidence]),
    }


def predict_single(model, image: np.ndarray, hand: np.ndarray | None = None,
                   device: str | None = None, is_hmlff: bool = False) -> Dict:
    torch, _ = _import_torch()
    import torchvision.transforms as T
    device = device or cfg.DEVICE
    x = _to_tensor(image, T, torch).unsqueeze(0).to(device)
    with torch.no_grad():
        if is_hmlff and hand is not None:
            h = torch.from_numpy(hand.astype(np.float32)).unsqueeze(0).to(device)
            logits, gates = model(x, h)
        else:
            logits = model(x)
            gates = None
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return {
        "class": cfg.INDEX_CLASS[pred],
        "label_idx": pred,
        "confidence": float(probs[pred]),
        "probs": probs,
        "gates": (gates.cpu().numpy()[0] if gates is not None else None),
    }


def _severity_from_conf(conf: float) -> str:
    # Map model confidence to a coarse severity bucket (for the table column).
    if conf > 0.9:
        return "Severe"
    if conf > 0.75:
        return "Moderate"
    if conf > 0.5:
        return "Low"
    return "None"


def _to_tensor(image, T, torch):
    import cv2
    img = image
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0
               else np.clip(img * 255, 0, 255).astype(np.uint8))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    x = rgb.astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x).permute(2, 0, 1)


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn