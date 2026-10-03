"""Tests for in-fold feature selectors and survival helpers."""

import numpy as np
import pandas as pd
import pytest
from sksurv.util import Surv

from src.models.selectors import StabilitySelector, TopKMutualInfo
from src.models.survival import paired_c_bootstrap, survival_cv, survival_factory, truncate


def _signal_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + 0.8 * X[:, 1] + rng.normal(scale=0.7, size=n) > 0.5).astype(int)
    return X, y


def test_topk_keeps_informative_features():
    X, y = _signal_data()
    sel = TopKMutualInfo(k=2).fit(X, y)
    assert set(np.flatnonzero(sel.get_support())) == {0, 1}
    assert sel.transform(X).shape == (len(y), 2)


def test_selectors_accept_survival_targets():
    X, y = _signal_data()
    ys = Surv.from_arrays(y.astype(bool), np.linspace(1, 10, len(y)))
    assert TopKMutualInfo(k=3).fit(X, ys).get_support().sum() == 3
    assert StabilitySelector(n_boot=10).fit(X, ys).get_support()[0]


def test_stability_selector_minimum_features():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 6)), rng.integers(0, 2, 200)
    sel = StabilitySelector(n_boot=10, threshold=0.99, min_features=3).fit(X, y)
    assert sel.get_support().sum() == 3


def test_truncate_censors_after_tau():
    y = Surv.from_arrays([True, True, False], [2.0, 12.0, 15.0])
    t = truncate(y, tau=10.0)
    assert t["event"].tolist() == [True, False, False]
    assert t["time"].tolist() == [2.0, 10.0, 10.0]


def test_survival_cv_and_paired_bootstrap():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"Kreatynina": rng.normal(120, 30, n), "Manifestacja_Nerki": rng.integers(0, 2, n).astype(float)})
    risk = (X["Kreatynina"] - 120) / 30
    time = rng.exponential(np.exp(-risk) * 5)
    event = time < 8
    y = Surv.from_arrays(event, np.minimum(time, 8) + 0.01)

    def fp(X_tr, y_tr, X_te):
        return survival_factory("gb_cox", list(X.columns))().fit(X_tr, y_tr).predict(X_te)

    res = survival_cv("gb", fp, X, y, n_repeats=1, verbose=False)
    assert res["summary"]["harrell_c_mean"] > 0.6
    same = paired_c_bootstrap(y, res["oof"], res["oof"], n_boot=20)
    assert same["delta_harrell_c"] == pytest.approx(0.0)
