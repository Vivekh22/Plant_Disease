"""
visualization.py
Generate and save publication-ready figures:
training/validation accuracy & loss, ROC, PR, confusion matrix, feature
importance, PCA / t-SNE / UMAP scatter plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config as cfg

log = logging.getLogger("hmlff_net.visualization")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT = True
except Exception:  # pragma: no cover
    _PLOT = False


def _save(fig, name: str) -> Path:
    path = cfg.FIGURE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", path)
    return path


def plot_history(history: Dict, model_name: str) -> Path:
    if not _PLOT:
        return Path()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_acc"], label="train")
    ax[0].plot(history["val_acc"], label="val")
    ax[0].set_title(f"{model_name} accuracy")
    ax[0].legend()
    ax[1].plot(history["train_loss"], label="train")
    ax[1].plot(history["val_loss"], label="val")
    ax[1].set_title(f"{model_name} loss")
    ax[1].legend()
    return _save(fig, f"history_{model_name}.png")


def plot_roc(roc_curves: List[dict], model_name: str) -> Path:
    if not _PLOT:
        return Path()
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, c in enumerate(roc_curves):
        ax.plot(c["fpr"], c["tpr"], label=f"class {i}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC - {model_name}")
    ax.legend(fontsize=7)
    return _save(fig, f"roc_{model_name}.png")


def plot_pr(pr_curves: List[dict], model_name: str) -> Path:
    if not _PLOT:
        return Path()
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, c in enumerate(pr_curves):
        ax.plot(c["recall"], c["precision"], label=f"class {i}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR - {model_name}")
    ax.legend(fontsize=7)
    return _save(fig, f"pr_{model_name}.png")


def plot_confusion(cm: List[List[int]], labels: List[str], model_name: str) -> Path:
    if not _PLOT:
        return Path()
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f"Confusion matrix - {model_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    return _save(fig, f"confusion_{model_name}.png")


def plot_feature_importance(importances: Dict[str, float], top_n: int = 30) -> Path:
    if not _PLOT:
        return Path()
    items = sorted(importances.items(), key=lambda kv: -kv[1])[:top_n]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(names, vals, color="teal")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features")
    return _save(fig, "feature_importance.png")


def plot_embedding(X: np.ndarray, y: np.ndarray, method: str,
                   title: str | None = None) -> Path:
    if not _PLOT:
        return Path()
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab20", s=10, alpha=0.7)
    ax.set_title(title or method)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, f"{method}.png")


def dim_reduce(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute PCA, t-SNE and UMAP embeddings and save each."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    embs = {}
    log.info("Computing PCA/t-SNE/UMAP ...")
    embs["pca"] = PCA(n_components=2).fit_transform(X)
    plot_embedding(embs["pca"], y, "pca", "PCA")
    try:
        embs["tsne"] = TSNE(n_components=2, init="pca", perplexity=30,
                            random_state=42).fit_transform(X)
        plot_embedding(embs["tsne"], y, "tsne", "t-SNE")
    except Exception as e:
        log.warning("t-SNE failed: %s", e)
    try:
        import umap
        embs["umap"] = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        plot_embedding(embs["umap"], y, "umap", "UMAP")
    except Exception as e:
        log.warning("UMAP failed: %s", e)
    return embs