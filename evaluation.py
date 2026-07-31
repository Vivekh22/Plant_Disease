"""
evaluation.py
Compute a comprehensive set of classification metrics and curves:
accuracy, precision, recall, specificity, F1, MCC, Cohen's kappa,
confusion matrix, ROC-AUC (OvR), PR curve.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

log = logging.getLogger("hmlff_net.evaluation")


class Evaluator:
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray | None = None,
                 num_classes: int | None = None) -> Dict:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, cohen_kappa_score, matthews_corrcoef,
            roc_auc_score, average_precision_score, precision_recall_curve,
            roc_curve,
        )
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = num_classes or int(y_true.max()) + 1
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        results: Dict = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "confusion_matrix": cm.tolist(),
        }
        # per-class specificity
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (fp + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
        results["specificity_macro"] = float(np.mean(tn / (tn + fp + 1e-9)))
        # ROC / PR
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            try:
                results["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            except Exception:
                results["roc_auc_ovr"] = float("nan")
            try:
                results["pr_auc_macro"] = float(
                    average_precision_score(y_true, y_prob, average="macro")
                )
            except Exception:
                results["pr_auc_macro"] = float("nan")
            # store curves for one-vs-rest macro
            fpr, tpr, roc_curves = {}, {}, []
            pr_curves = []
            for k in range(n):
                yt = (y_true == k).astype(int)
                ys = y_prob[:, k]
                f, t, _ = roc_curve(yt, ys)
                roc_curves.append({"fpr": f.tolist(), "tpr": t.tolist()})
                p, r, _ = precision_recall_curve(yt, ys)
                pr_curves.append({"precision": p.tolist(), "recall": r.tolist()})
            results["roc_curves"] = roc_curves
            results["pr_curves"] = pr_curves
        log.info("Eval done: acc=%.4f f1=%.4f mcc=%.4f kappa=%.4f",
                 results["accuracy"], results["f1_macro"],
                 results["mcc"], results["kappa"])
        return results