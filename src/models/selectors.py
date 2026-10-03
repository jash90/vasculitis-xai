"""
Feature selectors fitted inside a pipeline (i.e. inside each CV training fold).

Both accept a binary target or a scikit-survival structured array (the event
indicator is used), so the same selector works for classifiers and survival models.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _binary_target(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.names and "event" in y.dtype.names:
        return y["event"].astype(int)
    return y.astype(int)


class TopKMutualInfo(SelectorMixin, BaseEstimator):
    """Keep the k features with the highest mutual information with the outcome."""

    def __init__(self, k: int = 20, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.scores_ = mutual_info_classif(X, _binary_target(y), random_state=self.random_state)
        keep = np.argsort(self.scores_)[::-1][: min(self.k, X.shape[1])]
        self.mask_ = np.zeros(X.shape[1], bool)
        self.mask_[keep] = True
        return self

    def _get_support_mask(self):
        return self.mask_


class StabilitySelector(SelectorMixin, BaseEstimator):
    """Stability selection: L1-logistic on bootstrap half-samples, keep features
    selected in at least `threshold` of the fits (Meinshausen & Buehlmann 2010)."""

    def __init__(self, n_boot: int = 50, C: float = 0.05, threshold: float = 0.6, min_features: int = 5,
                 random_state: int = 42):
        self.n_boot = n_boot
        self.C = C
        self.threshold = threshold
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
        yb = _binary_target(y)
        rng = np.random.default_rng(self.random_state)
        counts = np.zeros(X.shape[1])
        for _ in range(self.n_boot):
            idx = rng.choice(len(yb), len(yb) // 2, replace=False)
            if len(np.unique(yb[idx])) < 2:
                continue
            lr = LogisticRegression(penalty="l1", C=self.C, solver="liblinear", class_weight="balanced")
            counts += lr.fit(X[idx], yb[idx]).coef_[0] != 0
        self.frequency_ = counts / self.n_boot
        mask = self.frequency_ >= self.threshold
        if mask.sum() < self.min_features:
            mask = np.zeros_like(mask)
            mask[np.argsort(self.frequency_)[::-1][: self.min_features]] = True
        self.mask_ = mask
        return self

    def _get_support_mask(self):
        return self.mask_
