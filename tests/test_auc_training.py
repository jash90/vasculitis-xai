"""
Tests for the AUC-first offline training path.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn

from src.models.auc_training import (
    build_auc_model_specs,
    evaluate_binary_classifier,
    prepare_auc_train_test,
)


def test_prepare_auc_train_test_excludes_identifier_and_target_columns(tmp_path):
    rng = np.random.default_rng(42)
    n_rows = 80

    data = pd.DataFrame(
        {
            "Kod": [f"PAT-{i:03d}" for i in range(n_rows)],
            "Patient_ID": np.arange(10_000, 10_000 + n_rows),
            "feature_a": rng.normal(size=n_rows),
            "feature_b": rng.normal(size=n_rows),
            "feature_c": rng.normal(size=n_rows),
            "feature_d": rng.normal(size=n_rows),
            "Zgon": np.tile([0, 1], n_rows // 2),
        }
    )
    csv_path = tmp_path / "patients.csv"
    data.to_csv(csv_path, sep="|", index=False)

    prepared = prepare_auc_train_test(csv_path, n_features=3, random_state=42)

    assert "Kod" not in prepared.feature_names
    assert "Patient_ID" not in prepared.feature_names
    assert "Zgon" not in prepared.feature_names
    assert prepared.X_train.shape[1] == 3
    assert prepared.X_test.shape[1] == 3


def test_build_auc_model_specs_includes_table_leaders():
    specs = build_auc_model_specs(random_state=42, scale_pos_weight=3.0)

    assert {"catboost", "stacking", "svm"}.issubset(specs)
    assert specs["catboost"].get_param("allow_writing_files") is False
    assert specs["stacking"].stack_method == "predict_proba"


def test_auc_training_runtime_dependencies_are_declared():
    requirements = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()

    assert "catboost" in requirements.lower()
    assert f"scikit-learn=={sklearn.__version__}" in requirements


def test_evaluate_binary_classifier_returns_table_metrics():
    class FixedProbabilityModel:
        def predict_proba(self, X):
            probabilities = np.array([0.05, 0.2, 0.7, 0.8, 0.35, 0.9])
            return np.column_stack([1 - probabilities, probabilities])

    X = np.zeros((6, 2))
    y = np.array([0, 0, 1, 1, 0, 1])

    metrics = evaluate_binary_classifier(FixedProbabilityModel(), X, y)

    assert set(
        [
            "auc",
            "ap",
            "accuracy",
            "recall",
            "specificity",
            "precision",
            "f1",
            "brier",
        ]
    ).issubset(metrics)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert json.dumps(metrics)
