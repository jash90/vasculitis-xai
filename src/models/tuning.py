"""
Optuna hyperparameter search and nested cross-validation.

`tune` searches on the data it receives only (inner CV). `nested_cv` calls it
inside every outer training fold, so the outer estimate is not biased by tuning.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.evaluation_protocol import CVResult, binary_metrics, positive_proba
from src.models.model_zoo import make_factory

optuna.logging.set_verbosity(optuna.logging.WARNING)

SearchSpace = Callable[[optuna.Trial], Dict[str, Any]]

SEARCH_SPACES: Dict[str, SearchSpace] = {
    "logreg": lambda t: (
        lambda pen: {"penalty": pen, "C": t.suggest_float("C", 1e-3, 10, log=True),
                     **({"l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0)} if pen == "elasticnet" else {})}
    )(t.suggest_categorical("penalty", ["l2", "elasticnet"])),
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 5),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 20, log=True),
    },
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 31),
        "min_child_samples": t.suggest_int("min_child_samples", 10, 60),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 20, log=True),
    },
    "catboost": lambda t: {
        "iterations": t.suggest_int("iterations", 200, 800, step=100),
        "depth": t.suggest_int("depth", 3, 6),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 20, log=True),
    },
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 300, 800, step=100),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        "max_depth": t.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
    },
    "hist_gb": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": t.suggest_int("max_iter", 100, 500, step=50),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 4, 31),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 60),
        "l2_regularization": t.suggest_float("l2_regularization", 0.0, 10.0),
    },
}


def tune(
    kind: str,
    columns: Sequence[str],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_trials: int = 25,
    inner_folds: int = 3,
    random_state: int = 42,
    factory_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the best parameters found by TPE with inner stratified CV (ROC AUC)."""
    space = SEARCH_SPACES[kind]
    factory_kwargs = factory_kwargs or {}
    cv = StratifiedKFold(inner_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        model = make_factory(kind, columns, params=params, random_state=random_state, **factory_kwargs)()
        return float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    # rebuild the full parameter dict (conditional parameters included)
    return space(optuna.trial.FixedTrial(study.best_params))


def nested_cv(
    kind: str,
    columns: Sequence[str],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 2,
    n_trials: int = 25,
    random_state: int = 42,
    factory_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> CVResult:
    """Outer repeated CV; Optuna tuning inside each outer training fold."""
    y = np.asarray(y, dtype=int)
    oof = np.zeros((n_repeats, len(y)))
    rows, chosen = [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state + r)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            params = tune(kind, columns, X_tr, y_tr, n_trials=n_trials, random_state=random_state,
                          factory_kwargs=factory_kwargs)
            chosen.append(params)
            model = make_factory(kind, columns, params=params, random_state=random_state,
                                 **(factory_kwargs or {}))().fit(X_tr, y_tr)
            oof[r, test_idx] = positive_proba(model, X.iloc[test_idx])
        rows.append(binary_metrics(y, oof[r]))
    return CVResult(name=name or f"{kind}_nested", oof=oof, thresholds=np.full_like(oof, np.nan),
                    per_repeat=pd.DataFrame(rows), y=y, meta={"outer_fold_params": str(chosen)})
