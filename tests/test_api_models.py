"""
Integration tests against the real trained artifacts in models/saved
(skipped when the artifacts are not present, e.g. in a fresh clone).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = [ROOT / "models/saved/best_model.joblib", ROOT / "models/saved/survival_rsf.joblib",
             ROOT / "models/saved/models_metadata.json"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="trained artifacts not available")

PATIENT = {"wiek_rozpoznania": 70, "manifestacja_nerki": 1, "manifestacja_oddechowy": 1,
           "kreatynina": 400, "max_crp": 120, "liczba_zajetych_narzadow": 2}


@pytest.fixture(scope="module")
def client():
    import os

    os.chdir(ROOT)
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_model_info_uses_training_metadata(client):
    body = client.get("/model/info").json()
    assert body["training_date"] and body["details"]["classifiers"]["xgboost"]["holdout"]["roc_auc"] > 0.5
    assert body["performance_metrics"]["roc_auc"] == pytest.approx(
        body["details"]["classifiers"]["xgboost"]["holdout"]["roc_auc"])
    assert "survival" in body["details"]


def test_survival_prediction_is_monotone(client):
    body = client.post("/predict/survival", json=PATIENT).json()
    assert 0 <= body["risk_1y"] <= body["risk_3y"] <= body["risk_5y"] <= 1
    curve = [p["survival"] for p in body["survival_curve"]]
    assert curve[0] == 1.0 and all(a >= b for a, b in zip(curve, curve[1:]))
    assert body["metrics"]["holdout_harrell_c"] > 0.5


def test_unknown_values_are_accepted(client):
    body = client.post("/predict", json={"wiek_rozpoznania": 50, "manifestacja_nerki": None,
                                         "kreatynina": None, "biopsja_wynik": None}).json()
    assert 0 <= body["probability"] <= 1


def test_batch_factors_are_patient_specific(client):
    patients = [PATIENT, {"wiek_rozpoznania": 25, "manifestacja_nos_ucho_gardlo": 1, "kreatynina": 70}]
    body = client.post("/predict/batch", json={"patients": patients, "include_risk_factors": True,
                                               "top_n_factors": 3}).json()
    f0 = [f["feature"] for f in body["results"][0]["top_risk_factors"]]
    f1 = [f["feature"] for f in body["results"][1]["top_risk_factors"]]
    assert f0 and f1 and f0 != f1


def test_global_importance_comes_from_model(client):
    body = client.get("/model/global-importance").json()
    assert "Permutation" in body["method"]
    assert sum(body["feature_importance"].values()) == pytest.approx(1.0, abs=1e-6)


def test_agent_skip_means_unknown_and_organ_count_is_derived(client):
    d = client.post("/agent/chat", json={"message": "start", "current_step": 0}).json()
    answers = ["60", "nie wiem"] + ["tak", "tak"] + ["nie"] * 6 + ["nie wiem"] * 3 + ["nie"] * 3
    for a in answers:
        d = client.post("/agent/chat", json={"message": a, "collected_data": d["collected_data"],
                                             "current_step": d["current_step"], "phase": d["phase"]}).json()
    assert d["phase"] == "discussion"
    assert d["collected_data"]["opoznienie_rozpoznia"] is None
    assert d["collected_data"]["kreatynina"] is None
    assert d["prediction_data"] is not None
