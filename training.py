"""
training.py
End-to-end training infrastructure:
    - 7 backbone models (ResNet50, EfficientNetV2, DenseNet121, MobileNetV3,
      VisionTransformer, MobileViT, HMLFFNet) via timm.
    - 5-fold Stratified Cross-Validation.
    - Bayesian (Optuna) hyperparameter optimization.
    - Loss functions: CrossEntropy, Focal, Weighted (class-imbalance aware).
    - MixUp / CutMix augmentation during training.
    - Early stopping + best-checkpoint saving.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

from . import config as cfg
from . import feature_fusion as ff

log = logging.getLogger("hmlff_net.training")

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
# Losses
# ---------------------------------------------------------------------------
class FocalLoss:
    """Focal loss for class imbalance (torch tensors)."""

    def __init__(self, gamma: float = 2.0, alpha: np.ndarray | None = None):
        torch, nn = _import_torch()
        self.torch = torch
        self.gamma = gamma
        self.alpha = (torch.tensor(alpha, dtype=torch.float32)
                      if alpha is not None else None)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def __call__(self, logits, target):
        ce = self.ce(logits, target)
        p = self.torch.exp(-ce)
        loss = ((1 - p) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def make_loss(name: str, num_classes: int, class_counts: np.ndarray | None = None):
    torch, nn = _import_torch()
    if name == "focal":
        alpha = (class_counts.sum() / (num_classes * np.maximum(class_counts, 1))) if class_counts is not None else None
        return FocalLoss(gamma=2.0, alpha=alpha)
    if name == "weighted":
        weight = (torch.tensor(class_counts.sum() / (num_classes * np.maximum(class_counts, 1)),
                               dtype=torch.float32)
                  if class_counts is not None else None)
        return nn.CrossEntropyLoss(weight=weight.to(torch.float32) if weight is not None else None,
                                   label_smoothing=cfg.TRAIN_CFG.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=cfg.TRAIN_CFG.label_smoothing)


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------
def build_classifier(model_name: str, num_classes: int, dropout: float = 0.3):
    """Build a timm-backed classifier with a custom dropout head."""
    torch, nn = _import_torch()
    import timm

    if model_name == "ResNet50":
        backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        in_dim = backbone.num_features
    elif model_name == "EfficientNetV2":
        backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0)
        in_dim = backbone.num_features
    elif model_name == "DenseNet121":
        backbone = timm.create_model("densenet121", pretrained=True, num_classes=0)
        in_dim = backbone.num_features
    elif model_name == "MobileNetV3":
        backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0)
        in_dim = backbone.num_features
    elif model_name == "VisionTransformer":
        backbone = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True, num_classes=0)
        in_dim = backbone.num_features
    elif model_name == "MobileViT":
        backbone = timm.create_model("mobilevit_s.cvnets_in1k", pretrained=True, num_classes=0)
        in_dim = backbone.num_features
    else:
        raise ValueError(f"Unknown model {model_name}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, num_classes),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    return Net()


def build_hmlff_net(num_classes: int, hand_dim: int, dropout: float = 0.3):
    """Proposed hybrid model: EfficientNetV2 + ViT + fusion -> classifier."""
    torch, nn = _import_torch()
    import timm

    cnn = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0)
    vit = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True, num_classes=0)
    cnn_dim, vit_dim = cnn.num_features, vit.num_features
    fusion = ff.build_fusion_module(hand_dim, cnn_dim, vit_dim, cfg.FUSION_DIM)

    class HMLFFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = cnn
            self.vit = vit
            self.fusion = fusion
            self.classifier = nn.Sequential(
                nn.LayerNorm(cfg.FUSION_DIM),
                nn.Dropout(dropout),
                nn.Linear(cfg.FUSION_DIM, num_classes),
            )

        def forward(self, x, hand=None):
            c = self.cnn(x)
            v = self.vit(x)
            if hand is None:
                hand = torch.zeros(x.size(0), self.fusion.proj_hand.in_features,
                                   device=x.device)
            fused, gates = self.fusion(hand, c, v)
            return self.classifier(fused), gates

    return HMLFFNet()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def make_optimizer(params, name: str, lr: float, weight_decay: float):
    torch, nn = _import_torch()
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Training loop for one fold
# ---------------------------------------------------------------------------
def train_one_model(
    model_name: str,
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    loss_name: str = "focal",
    class_counts: np.ndarray | None = None,
    hand_dim: int | None = None,
    use_mixup: bool = True,
    device: str | None = None,
) -> Dict:
    torch, nn = _import_torch()
    device = device or cfg.DEVICE
    is_hmlff = model_name == "HMLFFNet"

    model = (build_hmlff_net(num_classes, hand_dim, dropout) if is_hmlff
             else build_classifier(model_name, num_classes, dropout)).to(device)
    optimizer = make_optimizer(model.parameters(), optimizer_name, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = make_loss(loss_name, num_classes, class_counts)

    best_acc, best_state, patience, best_epoch = 0.0, None, 0, -1
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        tl, correct, total = 0.0, 0, 0
        for batch in train_loader:
            imgs, labels = batch["image"].to(device), batch["label"].to(device)
            hand = batch.get("hand")
            if use_mixup and np.random.rand() < 0.5 and not is_hmlff:
                imgs, y_a, y_b, lam = _mixup_torch(imgs, labels)
                optimizer.zero_grad()
                out = model(imgs)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                optimizer.zero_grad()
                if is_hmlff and hand is not None:
                    out, _ = model(imgs, hand.to(device))
                else:
                    out = model(imgs)
                loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()

        # Validation
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, is_hmlff)
        train_loss, train_acc = tl / max(total, 1), correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        log.info("[%s] epoch %02d train_acc=%.4f val_acc=%.4f train_loss=%.4f val_loss=%.4f",
                 model_name, epoch, train_acc, val_acc, train_loss, val_loss)

        if val_acc > best_acc:
            best_acc, best_epoch, best_state, patience = val_acc, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg.TRAIN_CFG.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_acc": best_acc,
            "best_epoch": best_epoch}


def _evaluate(model, loader, criterion, device, is_hmlff):
    torch, nn = _import_torch()
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch["image"].to(device), batch["label"].to(device)
            hand = batch.get("hand")
            if is_hmlff and hand is not None:
                out, _ = model(imgs, hand.to(device))
            else:
                out = model(imgs)
            loss += criterion(out, labels).item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return loss / max(total, 1), correct / max(total, 1)


def _mixup_torch(images, labels, alpha: float = 0.2):
    torch, nn = _import_torch()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0))
    return images * lam + images[idx] * (1 - lam), labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Cross validation + Bayesian optimization
# ---------------------------------------------------------------------------
def cross_validate(model_name: str, dataset, num_classes: int,
                   n_splits: int = 5, **train_kwargs) -> Dict:
    """Stratified k-fold cross validation; returns per-fold results."""
    from sklearn.model_selection import StratifiedKFold
    torch, nn = _import_torch()
    labels = np.array([d["label_idx"] for d in dataset])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.SEED)
    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        log.info("=== %s fold %d/%d ===", model_name, fold + 1, n_splits)
        tr_ds = [dataset[i] for i in tr]
        va_ds = [dataset[i] for i in va]
        tr_loader = _make_loader(tr_ds, train=True)
        va_loader = _make_loader(va_ds, train=False)
        res = train_one_model(model_name, tr_loader, va_loader,
                               num_classes, **train_kwargs)
        fold_results.append({"best_acc": res["best_acc"],
                             "history": res["history"]})
    accs = [r["best_acc"] for r in fold_results]
    return {"fold_results": fold_results, "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs))}


def _make_loader(ds, train: bool, batch_size: int | None = None):
    torch, nn = _import_torch()
    from torch.utils.data import DataLoader
    bs = batch_size or cfg.TRAIN_CFG.batch_size
    return DataLoader(ds, batch_size=bs, shuffle=train,
                     num_workers=cfg.TRAIN_CFG.num_workers, pin_memory=True)


def bayesian_optimize(model_name: str, dataset, num_classes: int,
                      n_trials: int = 15) -> Dict:
    """Optuna search over lr, dropout, batch_size, weight_decay, epochs."""
    import optuna
    from sklearn.model_selection import StratifiedKFold

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "epochs": trial.suggest_int("epochs", 10, 40, step=5),
            "optimizer_name": trial.suggest_categorical("optimizer_name",
                                                         ["adam", "adamw", "sgd"]),
            "loss_name": trial.suggest_categorical("loss_name",
                                                   ["cross_entropy", "focal", "weighted"]),
        }
        labels = np.array([d["label_idx"] for d in dataset])
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SEED)
        tr, va = next(skf.split(np.zeros(len(labels)), labels))
        tr_loader = _make_loader([dataset[i] for i in tr], train=True)
        va_loader = _make_loader([dataset[i] for i in va], train=False)
        res = train_one_model(model_name, tr_loader, va_loader, num_classes, **params)
        return res["best_acc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info("Best params: %s (acc=%.4f)", study.best_params, study.best_value)
    return {"best_params": study.best_params, "best_value": study.best_value}