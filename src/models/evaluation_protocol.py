"""
Leakage-safe evaluation protocol.

Every model is passed as a factory returning an *unfitted* pipeline, so all
preprocessing (imputation, encoding, scaling, resampling) is fitted inside each
training fold. Metrics are computed on out-of-fold (OOF) predictions; decision
thresholds are chosen on training-fold data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ModelFactory = Callable[[], BaseEstimator]

def positive_proba(model: BaseEstimator, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal: a=0, b=1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    slope = float(lr.coef_[0, 0])
    # Calibration-in-the-large: intercept with logit(p) as a fixed offset (Newton steps).
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + logit[:, 0])))
        step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12)
        a += step
        if abs(step) < 1e-8:
            break
    return slope, float(a)


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.85) -> float:
    """Highest threshold whose sensitivity on (y, p) is >= target."""
    positives = np.sort(p[y == 1])[::-1]
    if len(positives) == 0:
        return 0.5
    k = int(np.ceil(target * len(positives)))
    return float(positives[min(k, len(positives)) - 1])


def threshold_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Metrics of hard 0/1 predictions."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
    }


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
    if threshold is not None:
        out.update(threshold_metrics(y, (p >= threshold).astype(int)))
    return out


@dataclass
class CVResult:
    name: str
    oof: np.ndarray  # shape (n_repeats, n_samples)
    thresholds: np.ndarray  # per-sample threshold of the fold that predicted it
    per_repeat: pd.DataFrame
    y: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        row: Dict[str, float] = {"model": self.name}
        for col in self.per_repeat.columns:
            row[f"{col}_mean"] = float(self.per_repeat[col].mean())
            row[f"{col}_sd"] = float(self.per_repeat[col].std(ddof=1)) if len(self.per_repeat) > 1 else 0.0
        lo, hi = bootstrap_ci(self.y, self.oof.mean(axis=0), roc_auc_score)
        row["roc_auc_ci_low"], row["roc_auc_ci_high"] = lo, hi
        row.update(self.meta)
        return row


def repeated_cv(
    factory: ModelFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    name: str = "model",
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    sensitivity_target: Optional[float] = 0.85,
    inner_threshold_folds: int = 3,
) -> CVResult:
    """Repeated stratified K-fold CV with fold-internal threshold selection."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    oof = np.zeros((n_repeats, n))
    thr = np.full((n_repeats, n), np.nan)
    rows: List[Dict[str, float]] = []

    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            model = factory().fit(X_tr, y_tr)
            oof[r, test_idx] = positive_proba(model, X.iloc[test_idx])
            if sensitivity_target is not None:
                inner = StratifiedKFold(inner_threshold_folds, shuffle=True, random_state=random_state)
                p_inner = cross_val_predict(factory(), X_tr, y_tr, cv=inner, method="predict_proba")[:, 1]
                thr[r, test_idx] = threshold_for_sensitivity(y_tr, p_inner, sensitivity_target)
        metrics = binary_metrics(y, oof[r])
        if sensitivity_target is not None:
            # each prediction is compared with the threshold of the fold that produced it
            metrics.update(threshold_metrics(y, (oof[r] >= thr[r]).astype(int)))
        rows.append(metrics)

    return CVResult(name=name, oof=oof, thresholds=thr, per_repeat=pd.DataFrame(rows), y=y)


def grouped_cv(
    factory: ModelFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    *,
    min_group_size: int = 20,
    name: str = "model",
) -> Dict[str, object]:
    """Leave-one-centre-out: small centres are pooled into one 'other' group."""
    y = np.asarray(y, dtype=int)
    groups = pd.Series(np.asarray(groups), index=X.index)
    sizes = groups.value_counts()
    g = groups.where(groups.map(sizes) >= min_group_size, "other").to_numpy()
    oof = np.zeros(len(y))
    per_centre = []
    for centre in np.unique(g):
        test = g == centre
        model = factory().fit(X.loc[~test], y[~test])
        oof[test] = positive_proba(model, X.loc[test])
        auc = roc_auc_score(y[test], oof[test]) if len(np.unique(y[test])) == 2 else np.nan
        per_centre.append({"centre": centre, "n": int(test.sum()), "deaths": int(y[test].sum()), "roc_auc": auc})
    return {
        "name": name,
        "pooled_roc_auc": float(roc_auc_score(y, oof)),
        "pooled": binary_metrics(y, oof),
        "per_centre": pd.DataFrame(per_centre),
        "oof": oof,
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Stratified percentile bootstrap CI."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y, dtype=int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values[b] = metric(y[idx], p[idx])
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """DeLong test for two correlated ROC AUCs (Sun & Xu 2014 fast algorithm)."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    preds = np.vstack([p1, p2])[:, order]
    n = preds.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}
