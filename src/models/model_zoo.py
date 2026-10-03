"""
Unfitted model pipelines for the at-diagnosis models.

Every factory returns a pipeline that takes the recoded cohort DataFrame
(NaN = missing) and does imputation / encoding / scaling itself, so it can be
cross-validated without leakage and served directly.

Missingness is imputed (no missing-indicator features and no native-NaN
routing in boosted trees) because in this registry missingness itself
correlates with death (incomplete records of deceased patients).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.feature_sets import CATEGORICAL_FEATURES

POS_WEIGHT = 710 / 189  # negatives / positives in the full cohort


def make_preprocessor(columns: Sequence[str], *, imputer: str = "median", scale: bool = True) -> ColumnTransformer:
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in columns if c not in CATEGORICAL_FEATURES]
    if imputer == "iterative":
        num_imp = IterativeImputer(max_iter=10, random_state=42, skip_complete=True)
    else:
        num_imp = SimpleImputer(strategy=imputer)
    num_steps = [("impute", num_imp)] + ([("scale", StandardScaler())] if scale else [])
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _estimator(kind: str, params: Optional[Dict[str, Any]], balance: str, random_state: int):
    params = dict(params or {})
    weighted = balance == "weights"
    if kind == "logreg":
        return LogisticRegression(
            C=params.pop("C", 0.1), penalty=params.pop("penalty", "l2"),
            l1_ratio=params.pop("l1_ratio", None),
            solver="saga", max_iter=5000, class_weight="balanced" if weighted else None,
            random_state=random_state, **params,
        )
    if kind == "random_forest":
        defaults = dict(n_estimators=500, min_samples_leaf=5, max_features="sqrt", n_jobs=-1)
        defaults.update(params)
        return RandomForestClassifier(class_weight="balanced_subsample" if weighted else None,
                                      random_state=random_state, **defaults)
    if kind == "hist_gb":
        defaults = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)
        defaults.update(params)
        return HistGradientBoostingClassifier(class_weight="balanced" if weighted else None,
                                              random_state=random_state, **defaults)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        defaults = dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5,
                        subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0)
        defaults.update(params)
        return XGBClassifier(scale_pos_weight=POS_WEIGHT if weighted else 1.0, eval_metric="logloss",
                             random_state=random_state, n_jobs=2, **defaults)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        defaults = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)
        defaults.update(params)
        return LGBMClassifier(class_weight="balanced" if weighted else None, random_state=random_state,
                              n_jobs=2, verbose=-1, **defaults)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        defaults = dict(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5.0)
        defaults.update(params)
        return CatBoostClassifier(auto_class_weights="Balanced" if weighted else None, random_seed=random_state,
                                  verbose=False, allow_writing_files=False, thread_count=2, **defaults)
    if kind == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        defaults = dict(interactions=5, max_bins=64, outer_bags=8)
        defaults.update(params)
        return ExplainableBoostingClassifier(random_state=random_state, n_jobs=2, **defaults)
    raise ValueError(f"Unknown model kind: {kind}")


MODEL_KINDS = ("logreg", "random_forest", "hist_gb", "xgboost", "lightgbm", "catboost", "ebm")


def make_factory(
    kind: str,
    columns: Sequence[str],
    *,
    params: Optional[Dict[str, Any]] = None,
    imputer: str = "median",
    balance: str = "weights",
    select_l1_C: Optional[float] = None,
    calibration: Optional[str] = None,
    selector: Optional[Callable[[], Any]] = None,
    random_state: int = 42,
) -> Callable[[], Any]:
    """Return a zero-arg factory producing a fresh unfitted pipeline.

    balance: 'weights' (class weights), 'smote' (SMOTE inside the pipeline) or 'none'.
    select_l1_C: if set, L1-logistic feature selection is fitted inside the pipeline.
    calibration: 'sigmoid' / 'isotonic' wraps the pipeline in 5-fold CalibratedClassifierCV.
    selector: factory of a feature selector fitted in fold (e.g. src.models.selectors).
    """
    columns = list(columns)

    def factory():
        scale = kind == "logreg" or select_l1_C is not None
        steps = [("prep", make_preprocessor(columns, imputer=imputer, scale=scale))]
        if select_l1_C is not None:
            l1 = LogisticRegression(penalty="l1", C=select_l1_C, solver="liblinear",
                                    class_weight="balanced", random_state=random_state)
            steps.append(("select", SelectFromModel(l1)))
        if selector is not None:
            steps.append(("select_custom", selector()))
        if balance == "smote":
            steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=5)))
        steps.append(("model", _estimator(kind, params, balance, random_state)))
        pipe = ImbPipeline(steps) if balance == "smote" else Pipeline(steps)
        if calibration:
            return CalibratedClassifierCV(pipe, method=calibration, cv=5)
        return pipe

    return factory


def make_stacking_factory(
    base: Dict[str, Callable[[], Any]], *, random_state: int = 42
) -> Callable[[], Any]:
    """Stacking on out-of-fold base predictions (internal 5-fold), logistic meta-model."""

    def factory():
        return StackingClassifier(
            estimators=[(name, f()) for name, f in base.items()],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=5,
            stack_method="predict_proba",
            n_jobs=1,
        )

    return factory


def default_factories(columns: Iterable[str], **kwargs) -> Dict[str, Callable[[], Any]]:
    return {kind: make_factory(kind, list(columns), **kwargs) for kind in MODEL_KINDS}
