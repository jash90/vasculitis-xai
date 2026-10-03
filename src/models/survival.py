"""
Survival-analysis helpers (scikit-survival) shared by the experiment scripts.

All models are pipelines on the recoded cohort DataFrame (preprocessing fitted in
fold). Evaluation is administratively censored at TAU years because IPCW-based
metrics need overlapping follow-up between training and test folds.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from src.models.model_zoo import make_preprocessor

EVAL_TIMES = np.array([1.0, 3.0, 5.0])
TAU = 10.0


# ----------------------------------------------------------------------------- models
def survival_factory(
    kind: str, cols: List[str], *, selector: Optional[Callable[[], Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Callable[[], Pipeline]:
    """Unfitted survival pipeline; `selector` (a factory) is inserted after preprocessing."""

    def factory():
        prep = make_preprocessor(cols, scale=True)
        if kind == "coxnet":
            est = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.05, n_alphas=30, max_iter=10000)
        elif kind == "rsf":
            est = RandomSurvivalForest(n_estimators=400, min_samples_leaf=10, max_features="sqrt",
                                       n_jobs=-1, random_state=42)
        elif kind == "gb_cox":
            est = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.03, max_depth=2,
                                                   subsample=0.8, random_state=42)
        else:
            raise ValueError(kind)
        if params:
            est.set_params(**params)
        steps = [("prep", prep)] + ([("select", selector())] if selector else []) + [("model", est)]
        return Pipeline(steps)

    return factory


class CoxnetAtAlpha:
    """Coxnet with alpha chosen inside the training fold (inner 3-fold Harrell C)."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        path = survival_factory("coxnet", self.cols)().fit(X, y)
        alphas = path.named_steps["model"].alphas_
        candidates = alphas[:: max(1, len(alphas) // 10)]
        inner = StratifiedKFold(3, shuffle=True, random_state=42)
        scores = []
        for a in candidates:
            s = []
            for tr, te in inner.split(X, y["event"]):
                m = survival_factory("coxnet", self.cols)()
                m.set_params(model__alphas=[a]).fit(X.iloc[tr], y[tr])
                s.append(concordance_index_censored(y[te]["event"], y[te]["time"], m.predict(X.iloc[te]))[0])
            scores.append(np.mean(s))
        self.alpha_ = float(candidates[int(np.argmax(scores))])
        self.model_ = survival_factory("coxnet", self.cols)().set_params(model__alphas=[self.alpha_]).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ------------------------------------------------------------------------- evaluation
def truncate(y: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Administrative censoring at tau for evaluation (IPCW needs overlapping follow-up)."""
    return Surv.from_arrays(y["event"] & (y["time"] <= tau), np.minimum(y["time"], tau))


def survival_cv(name: str, fit_predict: Callable, X: pd.DataFrame, y: np.ndarray, n_repeats: int,
                verbose: bool = True) -> Dict:
    """Repeated stratified CV; returns per-repeat metrics and mean OOF risk."""
    rows, oof_all = [], np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + r)
        aucs = []
        for tr, te in skf.split(X, y["event"]):
            risk = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
            oof_all[r, te] = risk
            y_tr_eval, y_te_eval = truncate(y[tr]), truncate(y[te])
            times = EVAL_TIMES[EVAL_TIMES < y_te_eval["time"].max()]
            auc_t, _ = cumulative_dynamic_auc(y_tr_eval, y_te_eval, risk, times)
            aucs.append(dict(zip([f"auc_{t:g}y" for t in times], auc_t)))
        risk = oof_all[r]
        row = {
            "harrell_c": concordance_index_censored(y["event"], y["time"], risk)[0],
            "uno_c": concordance_index_ipcw(truncate(y), truncate(y), risk, tau=TAU)[0],
            **pd.DataFrame(aucs).mean().to_dict(),
        }
        rows.append(row)
    per = pd.DataFrame(rows)
    summary = {"model": name, **{f"{k}_mean": per[k].mean() for k in per}, **{f"{k}_sd": per[k].std(ddof=1) for k in per}}
    if verbose:
        print(f"  {name:44s} Harrell C={summary['harrell_c_mean']:.3f}±{summary['harrell_c_sd']:.3f} "
              f"Uno C={summary['uno_c_mean']:.3f} AUC1y={summary.get('auc_1y_mean', np.nan):.3f} "
              f"AUC3y={summary.get('auc_3y_mean', np.nan):.3f} AUC5y={summary.get('auc_5y_mean', np.nan):.3f}")
    return {"summary": summary, "oof": oof_all.mean(0)}


def paired_c_bootstrap(y: np.ndarray, r1: np.ndarray, r2: np.ndarray, n_boot: int = 1000) -> Dict[str, float]:
    """Paired bootstrap of the Harrell C difference (r1 - r2) on the same patients."""
    rng = np.random.default_rng(42)
    c = lambda r, i: concordance_index_censored(y["event"][i], y["time"][i], r[i])[0]  # noqa: E731
    idx = np.arange(len(y))
    diff = c(r1, idx) - c(r2, idx)
    boots = np.array([c(r1, b) - c(r2, b) for b in (rng.choice(idx, len(idx)) for _ in range(n_boot))])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"delta_harrell_c": float(diff), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}


def sksurv_fp(factory):
    def fp(X_tr, y_tr, X_te):
        return factory().fit(X_tr, y_tr).predict(X_te)
    return fp


def coxnet_fp(cols):
    def fp(X_tr, y_tr, X_te):
        return CoxnetAtAlpha(cols).fit(X_tr, y_tr).predict(X_te)
    return fp
