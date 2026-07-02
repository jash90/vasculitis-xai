"""
AUC-first offline training utilities.

This path is intentionally separate from the 20-feature API model. It uses the
full clinical table for model comparison while keeping preprocessing artifacts
and feature order attached to the saved models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_TARGET = "Zgon"
DEFAULT_ID_COLUMNS = ("Kod", "ID", "id", "Patient_ID")
DEFAULT_N_FEATURES = 71


@dataclass
class AUCPreparedData:
    """Prepared train/test data and fitted preprocessing objects."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    raw_feature_names: List[str]
    dropped_identifier_columns: List[str]
    dropped_correlated_features: List[str]
    imputer: SimpleImputer
    selector: SelectKBest
    scaler: StandardScaler


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _drop_high_correlation_columns(
    X_train: pd.DataFrame,
    threshold: Optional[float],
) -> List[str]:
    if threshold is None or threshold <= 0 or X_train.shape[1] < 2:
        return []

    temp_imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        temp_imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    corr = X_imputed.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > threshold)]


def prepare_auc_train_test(
    data_path: str | Path,
    *,
    target_col: str = DEFAULT_TARGET,
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
    n_features: Optional[int] = DEFAULT_N_FEATURES,
    test_size: float = 0.2,
    random_state: int = 42,
    correlation_threshold: Optional[float] = 0.95,
) -> AUCPreparedData:
    """
    Build a leakage-safe train/test matrix for offline AUC optimisation.

    The split happens before imputing, selecting features, and scaling. Identifier
    columns are removed before any feature scoring to prevent patient-code leakage.
    """

    df = _normalise_columns(pd.read_csv(data_path, sep="|"))
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    id_set = set(id_columns)
    dropped_identifier_columns = [column for column in X.columns if column in id_set]
    if dropped_identifier_columns:
        X = X.drop(columns=dropped_identifier_columns)

    X = X.apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)

    keep_mask = y.notna()
    X = X.loc[keep_mask]
    y = y.loc[keep_mask].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    dropped_correlated_features = _drop_high_correlation_columns(
        X_train_raw,
        correlation_threshold,
    )
    if dropped_correlated_features:
        X_train_raw = X_train_raw.drop(columns=dropped_correlated_features)
        X_test_raw = X_test_raw.drop(columns=dropped_correlated_features)

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    k = X_train_imputed.shape[1] if n_features is None else min(n_features, X_train_imputed.shape[1])
    selector = SelectKBest(
        score_func=partial(mutual_info_classif, random_state=random_state),
        k=k,
    )
    X_train_selected = selector.fit_transform(X_train_imputed, y_train)
    X_test_selected = selector.transform(X_test_imputed)

    raw_feature_names = X_train_raw.columns.tolist()
    feature_names = X_train_raw.columns[selector.get_support()].tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return AUCPreparedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=feature_names,
        raw_feature_names=raw_feature_names,
        dropped_identifier_columns=dropped_identifier_columns,
        dropped_correlated_features=dropped_correlated_features,
        imputer=imputer,
        selector=selector,
        scaler=scaler,
    )


def _require_optional_estimators():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return CatBoostClassifier, LGBMClassifier, XGBClassifier


def build_auc_model_specs(
    *,
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
    include_stacking: bool = True,
) -> Dict[str, Any]:
    """Create fresh model instances tuned for the full-feature AUC path."""

    CatBoostClassifier, LGBMClassifier, XGBClassifier = _require_optional_estimators()
    scale_pos_weight = 1.0 if scale_pos_weight is None else float(scale_pos_weight)

    specs: Dict[str, Any] = {
        "catboost": CatBoostClassifier(
            iterations=406,
            depth=6,
            learning_rate=0.02092927267979338,
            l2_leaf_reg=7.67539109585377,
            border_count=162,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            allow_writing_files=False,
            random_seed=random_state,
            verbose=False,
        ),
        "svm": SVC(
            C=0.7055565297792288,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.014950816445843533,
            max_depth=3,
            num_leaves=54,
            min_child_samples=14,
            subsample=0.790544421197768,
            colsample_bytree=0.40646279954556636,
            reg_alpha=0.08687200661323258,
            reg_lambda=4.360324378447307e-06,
            is_unbalance=True,
            random_state=random_state,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.006279196181558339,
            min_child_weight=7,
            gamma=0.07693125296022804,
            subsample=0.8,
            colsample_bytree=0.5846252903648841,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=random_state,
            n_jobs=2,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.01702840441422188,
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=5,
            subsample=0.5586520980125771,
            random_state=random_state,
        ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        ),
    }

    if include_stacking:
        specs["stacking"] = StackingClassifier(
            estimators=[
                ("random_forest", specs["random_forest"]),
                ("catboost", specs["catboost"]),
                ("lightgbm", specs["lightgbm"]),
                ("xgboost", specs["xgboost"]),
                ("gradient_boosting", specs["gradient_boosting"]),
                ("svm", specs["svm"]),
            ],
            final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
            cv=5,
            stack_method="predict_proba",
            passthrough=False,
        )

    return specs


def _positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return expit(model.decision_function(X))
    return np.asarray(model.predict(X), dtype=float)


def evaluate_binary_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a fitted binary classifier with the metrics used in the thesis table."""

    y_proba = _positive_probability(model, X)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auc": float(roc_auc_score(y, y_proba)),
        "ap": float(average_precision_score(y, y_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y, y_proba)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def train_auc_models(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    model_keys: Optional[Iterable[str]] = None,
    n_features: int = DEFAULT_N_FEATURES,
    random_state: int = 42,
    include_stacking: bool = True,
) -> Dict[str, Any]:
    """Train full-feature models, evaluate them, and persist comparison artifacts."""

    prepared = prepare_auc_train_test(
        data_path,
        n_features=n_features,
        random_state=random_state,
    )
    negatives = int(np.sum(prepared.y_train == 0))
    positives = int(np.sum(prepared.y_train == 1))
    scale_pos_weight = negatives / positives if positives else 1.0

    specs = build_auc_model_specs(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        include_stacking=include_stacking,
    )
    selected_keys = list(model_keys) if model_keys is not None else list(specs.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for key in selected_keys:
        if key not in specs:
            raise ValueError(f"Unknown model key: {key}. Available: {sorted(specs)}")
        model = specs[key]
        model.fit(prepared.X_train, prepared.y_train)
        models[key] = model
        metrics[key] = evaluate_binary_classifier(model, prepared.X_test, prepared.y_test)
        joblib.dump(model, output_path / f"{key}_model.joblib")

    comparison = [
        {"model": key, **values}
        for key, values in sorted(metrics.items(), key=lambda item: item[1]["auc"], reverse=True)
    ]
    best_key = comparison[0]["model"]

    joblib.dump(models[best_key], output_path / "best_auc_model.joblib")
    joblib.dump(prepared.imputer, output_path / "auc_imputer.joblib")
    joblib.dump(prepared.selector, output_path / "auc_selector.joblib")
    joblib.dump(prepared.scaler, output_path / "auc_scaler.joblib")
    joblib.dump(prepared.X_train, output_path / "X_train_auc.joblib")
    joblib.dump(prepared.X_test, output_path / "X_test_auc.joblib")
    joblib.dump(prepared.y_train, output_path / "y_train_auc.joblib")
    joblib.dump(prepared.y_test, output_path / "y_test_auc.joblib")

    metadata = {
        "best_model": best_key,
        "n_features": len(prepared.feature_names),
        "feature_names": prepared.feature_names,
        "raw_feature_names": prepared.raw_feature_names,
        "dropped_identifier_columns": prepared.dropped_identifier_columns,
        "dropped_correlated_features": prepared.dropped_correlated_features,
        "train_samples": int(len(prepared.y_train)),
        "test_samples": int(len(prepared.y_test)),
        "train_positive": positives,
        "train_negative": negatives,
        "random_state": random_state,
    }

    (output_path / "feature_names_auc.json").write_text(
        json.dumps(prepared.feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "auc_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    (output_path / "auc_evaluation_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    (output_path / "auc_model_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    return {
        "prepared": prepared,
        "models": models,
        "metrics": metrics,
        "comparison": comparison,
        "metadata": metadata,
    }
