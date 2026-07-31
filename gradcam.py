"""
gradcam.py
Explainable-AI heatmap generation:
    - GradCAM, GradCAM++, ScoreCAM (pytorch-gradcam)
    - LIME image explanation
    - SHAP visualization (deep explainer)
All maps are saved to ``config.HEATMAP_DIR``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.gradcam")

_torch = None


def _torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _target_layer(model, name: str):
    """Pick a sensible target conv/last block for each backbone."""
    torch = _torch()
    bb = getattr(model, "backbone", None) or getattr(model, "cnn", None) or model
    # EfficientNetV2 / MobileNetV3 / DenseNet / MobileViT
    for cand in ["conv_head", "out_conv", "classifier_head",
                 "features.7", "features.8"]:
        mod = bb
        try:
            for part in cand.split("."):
                mod = getattr(mod, part)
            return mod
        except Exception:
            continue
    # fallback: last conv-like module
    for m in reversed(list(bb.modules())):
        if "Conv" in type(m).__name__:
            return m
    return bb


class GradCAMExplainer:
    def __init__(self, model, method: str = "gradcam"):
        torch = _torch()
        from pytorch_grad_cam import (
            GradCAM, GradCAMPlusPlus, ScoreCAM,
        )
        self.model = model
        self.target = _target_layer(model, "")
        method = method.lower()
        if method == "gradcam++":
            self.cam = GradCAMPlusPlus(model=model, target_layers=[self.target])
        elif method == "scorecam":
            self.cam = ScoreCAM(model=model, target_layers=[self.target])
        else:
            self.cam = GradCAM(model=model, target_layers=[self.target])

    def explain(self, image_tensor, target_class: Optional[int] = None):
        targets = None
        if target_class is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            targets = [ClassifierOutputTarget(target_class)]
        return self.cam(input_tensor=image_tensor, targets=targets)[0]


def save_heatmap(image: np.ndarray, mask: np.ndarray,
                 name: str, alpha: float = 0.5) -> Path:
    import cv2
    heat = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 255).astype(np.uint8) if image.max() > 1.0
                 else np.clip(image * 255, 0, 255).astype(np.uint8))
    if image.shape[:2] != heat.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)
    path = cfg.HEATMAP_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
    log.info("Saved heatmap %s", path)
    return path


def explain_all_methods(model, image_tensor, image_bgr, name_stem: str,
                        target_class: Optional[int] = None):
    """Run GradCAM, GradCAM++, ScoreCAM and save overlays."""
    out = {}
    for m in ["gradcam", "gradcam++", "scorecam"]:
        try:
            ex = GradCAMExplainer(model, m)
            mask = ex.explain(image_tensor, target_class)
            out[m] = save_heatmap(image_bgr, mask, f"{name_stem}_{m}.png")
        except Exception as e:
            log.warning("%s failed: %s", m, e)
    return out


def lime_explain(model, image_bgr: np.ndarray, name_stem: str,
                 num_samples: int = 200) -> Path:
    """LIME superpixel explanation."""
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        import cv2
        torch = _torch()
        explainer = lime_image.LimeImageExplainer()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0

        def _predict(imgs):
            with torch.no_grad():
                t = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).float()
                t = (t - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                out = model(t.to(cfg.DEVICE))
                return torch.softmax(out, dim=1).cpu().numpy()

        exp = explainer.explain_instance(rgb, _predict,
                                         top_labels=1, hide_color=0,
                                         num_samples=num_samples)
        temp, mask = exp.get_image_and_mask(exp.top_labels[0],
                                            positive_only=True, num_features=10,
                                            hide_rest=False)
        fig_img = mark_boundaries(temp, mask)
        from matplotlib import pyplot as plt
        plt.imsave(str(cfg.HEATMAP_DIR / f"{name_stem}_lime.png"),
                  (fig_img * 255).astype(np.uint8))
        return cfg.HEATMAP_DIR / f"{name_stem}_lime.png"
    except Exception as e:
        log.warning("LIME failed: %s", e)
        return Path()


def shap_visualize(model, background, samples, name_stem: str) -> Path:
    """SHAP DeepExplainer visualization for a small batch."""
    try:
        import shap
        torch = _torch()
        bg = torch.from_numpy(background).float().to(cfg.DEVICE)
        sp = torch.from_numpy(samples).float().to(cfg.DEVICE)
        e = shap.DeepExplainer(model, bg)
        sv = e.shap_values(sp)
        shap.image_plot(sv, samples.transpose(0, 2, 3, 1),
                        show=False)
        from matplotlib import pyplot as plt
        path = cfg.HEATMAP_DIR / f"{name_stem}_shap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        log.warning("SHAP viz failed: %s", e)
        return Path()