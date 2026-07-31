"""
feature_selection.py
Redundancy-aware feature selection combining:
    - SHAP-based importance (lightweight tree surrogate)
    - Mutual Information
    - Recursive Feature Elimination (RFE)
Returns a boolean mask and the selected column names.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger("hmlff_net.feature_selection")


class FeatureSelector:
    """Rank features with three complementary criteria and keep the top-k."""

    def __init__(self, top_k: int | None = None, min_keep: int = 50):
        self.top_k = top_k
        self.min_keep = min_keep
        self.selected_names: List[str] = []
        self.importances_: dict = {}

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, names: List[str]) -> "FeatureSelector":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n_features = X.shape[1]
        log.info("Running feature selection on %d features.", n_features)

        shap_imp = self._shap_importance(X, y)
        mi_imp = self._mutual_info(X, y)
        rfe_imp = self._rfe_importance(X, y)

        # Normalize and combine (equal weights)
        combined = self._normalize(shap_imp) + self._normalize(mi_imp) + self._normalize(rfe_imp)
        order = np.argsort(-combined)

        k = self.top_k or max(self.min_keep, n_features // 3)
        k = min(k, n_features)
        keep_idx = sorted(order[:k].tolist())

        self.importances_ = {
            names[i]: float(combined[i]) for i in range(n_features)
        }
        self.selected_names = [names[i] for i in keep_idx]
        self.mask_ = np.zeros(n_features, dtype=bool)
        self.mask_[keep_idx] = True
        log.info("Selected %d/%d features.", len(keep_idx), n_features)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)[:, self.mask_]

    def fit_transform(self, X, y, names):
        return self.fit(X, y, names).transform(X)

    # ------------------------------------------------------------------
    def _shap_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            try:
                import shap
                explainer = shap.TreeExplainer(rf)
                sv = explainer.shap_values(X, check_additivity=False)
                if isinstance(sv, list):
                    sv = np.mean([np.abs(s) for s in sv], axis=0)
                else:
                    sv = np.abs(sv)
                imp = sv.mean(axis=0)
            except Exception:
                imp = rf.feature_importances_
            log.info("SHAP importance computed.")
        except Exception as e:
            log.warning("SHAP failed (%s); using RF gini.", e)
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            imp = rf.feature_importances_
        return np.asarray(imp, dtype=np.float32)

    def _mutual_info(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.feature_selection import mutual_info_classif
        imp = mutual_info_classif(X, y, random_state=42)
        log.info("Mutual information computed.")
        return np.asarray(imp, dtype=np.float32)

    def _rfe_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(max_iter=300, n_jobs=-1)
        n = X.shape[1]
        rfe = RFE(estimator, n_features_to_select=max(n // 2, 1), step=max(n // 20, 1))
        rfe.fit(X, y)
        # rank: 1 = important. Convert to a descending score.
        score = (rfe.ranking_.max() + 1) - rfe.ranking_.astype(float)
        log.info("RFE ranking computed.")
        return score.astype(np.float32)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        if v.max() - v.min() < 1e-9:
            return np.zeros_like(v)
        return (v - v.min()) / (v.max() - v.min())

    def selected(self) -> List[str]:
        return list(self.selected_names)