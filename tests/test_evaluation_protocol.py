"""Tests for the leakage-safe evaluation protocol."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from src.models.evaluation_protocol import (
    binary_metrics,
    bootstrap_ci,
    calibration_slope_intercept,
    delong_test,
    grouped_cv,
    repeated_cv,
    threshold_for_sensitivity,
)
from src.models.model_zoo import make_factory


class RecordingModel(BaseEstimator, ClassifierMixin):
    """Remembers the row indices it was fitted on (to prove no test rows leak in)."""

    fitted_on: list = []

    def fit(self, X, y):
        RecordingModel.fitted_on.append(set(X.index))
        self.classes_ = np.array([0, 1])
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.clip(X["signal"].to_numpy(), 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"signal": np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1), "noise": rng.normal(size=n)})
    return X, y


def test_folds_never_train_on_their_test_rows():
    X, y = _data()
    RecordingModel.fitted_on = []
    res = repeated_cv(lambda: RecordingModel(), X, y, n_splits=5, n_repeats=1, sensitivity_target=None)
    outer = RecordingModel.fitted_on
    assert len(outer) == 5
    # every row is predicted exactly once and never by a model trained on it
    covered = set()
    for train_rows in outer:
        test_rows = set(X.index) - train_rows
        assert not (test_rows & train_rows)
        covered |= test_rows
    assert covered == set(X.index)
    assert res.per_repeat["roc_auc"].iloc[0] == pytest.approx(roc_auc_score(y, res.oof[0]))


def test_thresholds_come_from_training_folds():
    X, y = _data()
    res = repeated_cv(lambda: RecordingModel(), X, y, n_splits=5, n_repeats=2, sensitivity_target=0.85)
    assert not np.isnan(res.thresholds).any()
    assert {"sensitivity", "specificity", "balanced_accuracy"} <= set(res.per_repeat.columns)


def test_threshold_for_sensitivity():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.5, 0.7, 0.9])
    thr = threshold_for_sensitivity(y, p, 0.75)
    assert ((p >= thr) & (y == 1)).sum() / 4 >= 0.75
    assert thr == pytest.approx(0.5)


def test_delong_identical_and_different_predictions():
    X, y = _data(400, seed=1)
    p = X["signal"].to_numpy()
    same = delong_test(y, p, p)
    assert same["diff"] == pytest.approx(0.0)
    noisy = np.random.default_rng(3).uniform(size=len(y))
    res = delong_test(y, p, noisy)
    assert res["auc_1"] == pytest.approx(roc_auc_score(y, p))
    assert res["p_value"] < 0.01


def test_calibration_of_perfectly_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_bootstrap_ci_contains_point_estimate():
    X, y = _data(300, seed=2)
    p = X["signal"].to_numpy()
    lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo <= roc_auc_score(y, p) <= hi


def test_binary_metrics_keys():
    X, y = _data()
    m = binary_metrics(y, X["signal"].to_numpy(), 0.5)
    assert {"roc_auc", "pr_auc", "brier", "cal_slope", "sensitivity", "npv"} <= set(m)


def test_pipeline_factory_handles_missing_and_categoricals():
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({
        "Kreatynina": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.normal(120, 30, n)),
        "Pulsy": rng.integers(0, 4, n).astype(float),
        "Manifestacja_Nerki": rng.integers(0, 2, n).astype(float),
    })
    y = rng.integers(0, 2, n)
    for kind in ("logreg", "random_forest"):
        model = make_factory(kind, X.columns)().fit(X, y)
        assert model.predict_proba(X).shape == (n, 2)


def test_grouped_cv_reports_each_centre():
    X, y = _data(300)
    groups = pd.Series(np.repeat(["AAAAA", "BBBBB", "CCCCC"], 100))
    res = grouped_cv(lambda: RecordingModel(), X, y, groups, min_group_size=50)
    assert set(res["per_centre"]["centre"]) == {"AAAAA", "BBBBB", "CCCCC"}
