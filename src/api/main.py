"""
FastAPI aplikacja dla systemu XAI.

Główny plik API z endpointami dla predykcji,
wyjaśnień XAI i agenta konwersacyjnego.
"""

import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import json
import logging
import joblib

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Dodaj ścieżkę projektu
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .schemas import (
    PatientInput, PredictionOutput, SHAPExplanation, LIMEExplanation,
    PatientExplanation, ModelInfo, GlobalImportance, HealthCheckResponse,
    ChatRequest, ChatResponse, ExplanationRequest, PatientExplanationRequest,
    ComparisonResult, ErrorResponse, RiskLevel, XAIMethod, HealthLiteracyLevel,
    patient_to_array, get_risk_level_from_probability, patients_to_matrix,
    # Batch schemas
    BatchPatientInput, BatchPredictionOutput, BatchPatientResult,
    BatchSummary, BatchProcessingError, RiskFactorItem,
    DemoModeStatus, DemoModeRequest,
    # Agent schemas
    AgentPredictionData, AgentPredictionFactor,
    AgentConversationRequest, AgentConversationResponse,
    # Multi-model schemas
    MultiModelPredictionOutput, ModelPrediction,
    # DALEX/EBM schemas
    DALEXExplanation, EBMExplanation,
    # AUC offline model schemas
    AUCPredictionInput,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# APLIKACJA FASTAPI
# ============================================================================

app = FastAPI(
    title="Vasculitis XAI API",
    description="""
    API do predykcji śmiertelności w zapaleniu naczyń z wyjaśnieniami XAI.

    System wykorzystuje modele ML (XGBoost, Random Forest) wraz z metodami
    wyjaśnialnej sztucznej inteligencji (LIME, SHAP, DALEX, EBM) do:
    - Predykcji ryzyka zgonu
    - Generowania wyjaśnień dla klinicystów i pacjentów
    - Interaktywnej rozmowy o wynikach
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Stan aplikacji z wczytanymi modelami."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.lime_explainer = None
        self.shap_explainer = None
        self.rag_pipeline = None
        self.is_loaded = False

        # Konfiguracja trybu demo
        self.allow_demo = os.getenv("ALLOW_DEMO", "true").lower() == "true"
        self.force_api_mode = os.getenv("FORCE_API_MODE", "false").lower() == "true"

        if self.force_api_mode:
            self.allow_demo = False

        # Global feature importance cache
        self._global_importance_cache: Optional[Dict[str, float]] = None

    def set_demo_mode(self, enabled: bool):
        """Włącz/wyłącz tryb demo."""
        if self.force_api_mode and enabled:
            raise ValueError("Nie można włączyć trybu demo gdy FORCE_API_MODE=true")
        self.allow_demo = enabled
        logger.info(f"Tryb demo {'włączony' if enabled else 'wyłączony'}")

    def get_current_mode(self) -> str:
        """Pobierz aktualny tryb działania."""
        if self.is_loaded:
            return "api"
        elif self.allow_demo:
            return "demo"
        else:
            return "unavailable"

    def get_global_importance(self) -> Dict[str, float]:
        """Pobierz global feature importance (z cache)."""
        if self._global_importance_cache is None:
            self._global_importance_cache = {
                "Wiek": 0.15,
                "Manifestacja_Nerki": 0.12,
                "Zaostrz_Wymagajace_OIT": 0.11,
                "Liczba_Zajetych_Narzadow": 0.10,
                "Manifestacja_Sercowo-Naczyniowy": 0.09,
                "Kreatynina": 0.08,
                "Max_CRP": 0.07,
                "Dializa": 0.06,
                "Manifestacja_Zajecie_CSN": 0.05,
                "Plazmaferezy": 0.04,
                "Manifestacja_Neurologiczny": 0.03,
                "Manifestacja_Pokarmowy": 0.02,
                "Plec": 0.02,
                "Sterydy_Dawka_g": 0.02,
                "Czas_Sterydow": 0.01,
                "Powiklania_Serce/pluca": 0.02,
                "Powiklania_Infekcja": 0.02,
                "Wiek_rozpoznania": 0.01,
                "Opoznienie_Rozpoznia": 0.01
            }
        return self._global_importance_cache

    def load_models(self, model_path: str, feature_names_path: str):
        """Wczytaj modele i explainer'y."""
        import joblib

        try:
            # Wczytaj model
            self.model = joblib.load(model_path)
            logger.info(f"Model wczytany z {model_path}")

            # Wczytaj nazwy cech
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Wczytano {len(self.feature_names)} nazw cech")

            expected_features = getattr(self.model, "n_features_in_", None)
            if expected_features is None and hasattr(self.model, "get_booster"):
                try:
                    expected_features = self.model.get_booster().num_features()
                except Exception:
                    expected_features = None

            if expected_features is not None and int(expected_features) != len(self.feature_names):
                logger.warning(
                    "Model ma %s cech, ale feature_names.json ma %s. "
                    "Pomijam model i uruchamiam API w trybie demo.",
                    expected_features,
                    len(self.feature_names)
                )
                self.model = None
                self.feature_names = None
                self.is_loaded = False
                return False

            self.is_loaded = True

            # Inicjalizuj XAI explainery
            models_dir = Path(model_path).parent
            self._init_xai_explainers(models_dir)

            return True

        except Exception as e:
            logger.error(f"Błąd wczytywania modeli: {e}")
            return False

    def _init_xai_explainers(self, models_dir: Path):
        """Inicjalizuj SHAP i LIME explainery z prawdziwymi danymi."""
        import joblib
        from src.xai.shap_explainer import SHAPExplainer
        from src.xai.lime_explainer import LIMEExplainer

        try:
            X_background_path = models_dir / "X_background.joblib"
            X_train_path = models_dir / "X_train.joblib"

            if X_background_path.exists():
                X_background = joblib.load(X_background_path)
                self.shap_explainer = SHAPExplainer(
                    model=self.model,
                    X_background=X_background,
                    feature_names=self.feature_names
                )
                logger.info("SHAPExplainer zainicjalizowany z prawdziwymi danymi")
            else:
                logger.warning("X_background.joblib nie znaleziony — SHAP w trybie demo")

            if X_train_path.exists():
                X_train = joblib.load(X_train_path)
                self.lime_explainer = LIMEExplainer(
                    model=self.model,
                    X_train=X_train,
                    feature_names=self.feature_names
                )
                logger.info("LIMEExplainer zainicjalizowany z prawdziwymi danymi")
            else:
                logger.warning("X_train.joblib nie znaleziony — LIME w trybie demo")

        except Exception as e:
            logger.error(f"Błąd inicjalizacji XAI explainerów: {e}")
            self.shap_explainer = None
            self.lime_explainer = None


app_state = AppState()


def _load_model_by_key(model_key: str):
    """Załaduj model z dysku na podstawie klucza (xgboost/random_forest/lightgbm)."""
    if model_key == "xgboost" or model_key is None:
        if app_state.model is None:
            raise HTTPException(status_code=503, detail="Model nie jest wczytany")
        return app_state.model
    filename = _MODEL_FILES.get(model_key)
    if filename is None:
        raise HTTPException(status_code=400, detail=f"Nieznany model: {model_key}. Dostępne: xgboost, random_forest, lightgbm")
    model_path = Path("models/saved") / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_key} nie istnieje: {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd wczytywania modelu {model_key}: {e}")


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Zdarzenie uruchomienia aplikacji."""
    logger.info("Uruchamianie API Vasculitis XAI...")

    # Próba wczytania modeli
    model_path = os.getenv("MODEL_PATH", "models/saved/best_model.joblib")
    feature_names_path = os.getenv("FEATURE_NAMES_PATH", "models/saved/feature_names.json")

    if Path(model_path).exists() and Path(feature_names_path).exists():
        app_state.load_models(model_path, feature_names_path)
    else:
        logger.warning("Pliki modelu nie znalezione. API działa w trybie demo.")


@app.on_event("shutdown")
async def shutdown_event():
    """Zdarzenie zamknięcia aplikacji."""
    logger.info("Zamykanie API...")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_demo_prediction(patient: PatientInput) -> PredictionOutput:
    """Wygeneruj demo predykcję gdy model nie jest załadowany."""
    risk_score = 0.0

    risk_score += patient.liczba_zajetych_narzadow * 0.1

    if patient.manifestacja_nerki:
        risk_score += 0.15
    if patient.manifestacja_sercowo_naczyniowy:
        risk_score += 0.15
    if patient.manifestacja_zajecie_csn:
        risk_score += 0.2
    if patient.zaostrz_wymagajace_oit:
        risk_score += 0.25
    if patient.zaostrz_wymagajace_hospital:
        risk_score += 0.1
    if patient.plazmaferezy:
        risk_score += 0.1

    probability = min(max(risk_score, 0.05), 0.95)

    return PredictionOutput(
        probability=probability,
        risk_level=get_risk_level_from_probability(probability),
        prediction=1 if probability > 0.5 else 0,
        confidence_interval={"lower": max(0, probability - 0.1), "upper": min(1, probability + 0.1)}
    )


def get_demo_explanation(patient: PatientInput) -> dict:
    """Wygeneruj demo wyjaśnienie."""
    risk_factors = []
    protective_factors = []

    if patient.manifestacja_nerki:
        risk_factors.append({
            "feature": "Manifestacja_Nerki",
            "value": 1,
            "contribution": 0.12,
            "direction": "increases_risk"
        })

    if patient.zaostrz_wymagajace_oit:
        risk_factors.append({
            "feature": "Zaostrz_Wymagajace_OIT",
            "value": 1,
            "contribution": 0.2,
            "direction": "increases_risk"
        })

    if patient.liczba_zajetych_narzadow <= 2:
        protective_factors.append({
            "feature": "Liczba_Zajetych_Narzadow",
            "value": patient.liczba_zajetych_narzadow,
            "contribution": -0.08,
            "direction": "decreases_risk"
        })

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "base_value": 0.15
    }


def _auc_models_dir() -> Path:
    """Katalog artefaktów pełnego toru AUC."""
    return Path(os.getenv("AUC_MODELS_DIR", "models/saved/auc_all"))


@lru_cache(maxsize=4)
def _load_auc_metadata(models_dir: str) -> Dict[str, Any]:
    metadata_path = Path(models_dir) / "auc_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"Brak metadanych modelu AUC: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_auc_comparison(models_dir: str) -> List[Dict[str, Any]]:
    comparison_path = Path(models_dir) / "auc_model_comparison.json"
    if not comparison_path.exists():
        raise HTTPException(status_code=404, detail=f"Brak porównania modeli AUC: {comparison_path}")
    return json.loads(comparison_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_auc_preprocessors(models_dir: str):
    base = Path(models_dir)
    required = {
        "imputer": base / "auc_imputer.joblib",
        "selector": base / "auc_selector.joblib",
        "scaler": base / "auc_scaler.joblib",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Brak artefaktów preprocessingu AUC: {missing}")
    return (
        joblib.load(required["imputer"]),
        joblib.load(required["selector"]),
        joblib.load(required["scaler"]),
    )


@lru_cache(maxsize=32)
def _load_auc_prediction_model(models_dir: str, requested_model_key: Optional[str]):
    metadata = _load_auc_metadata(models_dir)
    comparison = _load_auc_comparison(models_dir)
    available_models = {str(row["model"]) for row in comparison if "model" in row}
    best_model = str(metadata.get("best_model", ""))
    model_key = best_model if requested_model_key in (None, "", "best") else str(requested_model_key)

    if model_key not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Nieznany model AUC: {model_key}. Dostępne: {sorted(available_models)}",
        )

    filename = "best_auc_model.joblib" if model_key == best_model else f"{model_key}_model.joblib"
    model_path = Path(models_dir) / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Brak modelu AUC: {model_path}")
    return model_key, joblib.load(model_path)


def _build_auc_raw_frame(features: Dict[str, Any], raw_feature_names: List[str]) -> pd.DataFrame:
    """Zbuduj jednowierszową ramkę w kolejności użytej przy treningu AUC."""
    row: Dict[str, float] = {}
    for feature_name in raw_feature_names:
        value = features.get(feature_name, np.nan)
        if value is None or value == "":
            row[feature_name] = np.nan
            continue
        try:
            row[feature_name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"Cecha AUC '{feature_name}' musi być liczbą albo null",
            )
    return pd.DataFrame([row], columns=raw_feature_names)


def _predict_positive_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        return expit(model.decision_function(X))
    return np.asarray(model.predict(X), dtype=float)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Strona główna API."""
    return {
        "name": "Vasculitis XAI API",
        "version": "1.0.0",
        "description": "API do predykcji śmiertelności w zapaleniu naczyń z wyjaśnieniami XAI",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Info"])
async def health_check():
    """Sprawdzenie stanu API."""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=app_state.is_loaded,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/models/auc", tags=["Info"])
async def auc_model_info():
    """Metadane pełnego, offline'owego treningu AUC."""
    models_dir = str(_auc_models_dir())
    metadata = _load_auc_metadata(models_dir)
    comparison = _load_auc_comparison(models_dir)

    return {
        "available": True,
        "models_dir": models_dir,
        "best_model": metadata.get("best_model"),
        "n_features": metadata.get("n_features"),
        "feature_names": metadata.get("feature_names", []),
        "raw_feature_names": metadata.get("raw_feature_names", []),
        "dropped_identifier_columns": metadata.get("dropped_identifier_columns", []),
        "dropped_correlated_features": metadata.get("dropped_correlated_features", []),
        "train_samples": metadata.get("train_samples"),
        "test_samples": metadata.get("test_samples"),
        "models": comparison,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Wykonaj predykcję ryzyka śmiertelności.

    Zwraca prawdopodobieństwo zgonu i poziom ryzyka dla pacjenta.
    """
    try:
        if app_state.is_loaded and app_state.model is not None:
            # Użyj prawdziwego modelu
            features = patient_to_array(patient, app_state.feature_names)
            X = np.array(features).reshape(1, -1)

            probability = app_state.model.predict_proba(X)[0, 1]
            prediction = int(probability > 0.5)

            return PredictionOutput(
                probability=float(probability),
                risk_level=get_risk_level_from_probability(probability),
                prediction=prediction
            )
        else:
            # Demo mode
            return get_demo_prediction(patient)

    except Exception as e:
        logger.error(f"Błąd predykcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/auc", tags=["Prediction"])
async def predict_auc(request: AUCPredictionInput):
    """
    Predykcja z pełnego modelu AUC trenowanego na wszystkich cechach klinicznych.

    Ten endpoint przyjmuje słownik cech, bo pełny model ma inny kontrakt wejścia
    niż 20-cechowy formularz używany przez standardowy `/predict`.
    """
    models_dir = str(_auc_models_dir())
    metadata = _load_auc_metadata(models_dir)
    raw_feature_names = metadata.get("raw_feature_names") or metadata.get("feature_names")
    if not raw_feature_names:
        raise HTTPException(status_code=500, detail="Metadane AUC nie zawierają listy cech")

    model_key, model = _load_auc_prediction_model(models_dir, request.model_key)
    imputer, selector, scaler = _load_auc_preprocessors(models_dir)

    raw_frame = _build_auc_raw_frame(request.features, raw_feature_names)
    X_imputed = imputer.transform(raw_frame)
    X_selected = selector.transform(X_imputed)
    X_scaled = scaler.transform(X_selected)

    probability = float(_predict_positive_probability(model, X_scaled)[0])
    prediction = int(probability > 0.5)
    missing_features = [
        name for name in raw_feature_names
        if name not in request.features or request.features.get(name) in (None, "")
    ]

    return {
        "probability": probability,
        "risk_level": get_risk_level_from_probability(probability),
        "prediction": prediction,
        "model_key": model_key,
        "n_features": int(metadata.get("n_features", X_scaled.shape[1])),
        "raw_input_features": len(raw_feature_names),
        "missing_features_count": len(missing_features),
        "missing_features_sample": missing_features[:20],
    }


# ============================================================================
# MULTI-MODEL PREDICTION
# ============================================================================

_MODEL_FILES = {
    "xgboost": "best_model.joblib",
    "random_forest": "random_forest_model.joblib",
    "lightgbm": "lightgbm_model.joblib",
}

_MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}


@app.post("/predict/all", response_model=MultiModelPredictionOutput, tags=["Prediction"])
async def predict_all_models(patient: PatientInput):
    """
    Predykcja ryzyka ze wszystkich dostępnych modeli.

    Zwraca wyniki XGBoost, Random Forest i LightGBM oraz średnią ensemble.
    """

    models_dir = Path("models/saved")
    features = patient_to_array(patient, app_state.feature_names)
    X = np.array(features).reshape(1, -1)

    results: List[ModelPrediction] = []
    probabilities: List[float] = []

    for model_key, filename in _MODEL_FILES.items():
        model_path = models_dir / filename
        if not model_path.exists():
            continue
        try:
            model = joblib.load(model_path)
            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob > 0.5)
            results.append(ModelPrediction(
                model_name=_MODEL_LABELS[model_key],
                probability=prob,
                risk_level=get_risk_level_from_probability(prob),
                prediction=pred,
            ))
            probabilities.append(prob)
        except Exception as e:
            logger.warning(f"Nie udało się wczytać modelu {model_key}: {e}")
            continue

    if not probabilities:
        raise HTTPException(status_code=500, detail="Brak dostępnych modeli")

    ensemble_prob = sum(probabilities) / len(probabilities)

    return MultiModelPredictionOutput(
        models=results,
        ensemble_probability=ensemble_prob,
        ensemble_risk_level=get_risk_level_from_probability(ensemble_prob),
        primary_model="XGBoost",
    )


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def batch_predict_vectorized(X: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch prediction using numpy.

    Args:
        X: Feature matrix (n_patients, n_features)
        model: Trained model with predict_proba

    Returns:
        Tuple of (probabilities, predictions)
    """
    probabilities = model.predict_proba(X)[:, 1]  # Positive class
    predictions = (probabilities > 0.5).astype(int)
    return probabilities, predictions


def get_batch_risk_factors(
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 3
) -> List[List[RiskFactorItem]]:
    """
    Extract top risk factors for each patient using global importance.
    Fast extraction for batch processing.
    """
    global_importance = app_state.get_global_importance()
    n_patients = X.shape[0]
    results = []

    # Feature name to index mapping
    feature_idx_map = {name: i for i, name in enumerate(feature_names)}

    # Thresholds for determining risk direction
    RISK_THRESHOLDS = {
        "Wiek": 60,
        "Kreatynina": 150,
        "Max_CRP": 50,
        "Liczba_Zajetych_Narzadow": 3,
    }

    BINARY_RISK_FEATURES = {
        "Manifestacja_Nerki", "Manifestacja_Sercowo-Naczyniowy",
        "Manifestacja_Zajecie_CSN", "Zaostrz_Wymagajace_OIT",
        "Dializa", "Plazmaferezy", "Manifestacja_Neurologiczny",
        "Manifestacja_Pokarmowy", "Powiklania_Serce/pluca", "Powiklania_Infekcja"
    }

    for i in range(n_patients):
        patient_factors = []

        for feature_name, importance in global_importance.items():
            if feature_name not in feature_idx_map:
                continue

            idx = feature_idx_map[feature_name]
            value = float(X[i, idx])

            # Determine direction based on feature type and value
            if feature_name in BINARY_RISK_FEATURES:
                direction = "increases_risk" if value == 1 else "decreases_risk"
            elif feature_name in RISK_THRESHOLDS:
                threshold = RISK_THRESHOLDS[feature_name]
                direction = "increases_risk" if value > threshold else "decreases_risk"
            else:
                direction = "neutral"

            patient_factors.append(RiskFactorItem(
                feature=feature_name,
                value=value,
                importance=importance,
                direction=direction
            ))

        # Sort by importance and take top N
        patient_factors.sort(key=lambda x: x.importance, reverse=True)
        results.append(patient_factors[:top_n])

    return results


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(request: BatchPatientInput):
    """
    Batch prediction dla wielu pacjentów.

    Używa vectorized numpy dla wysokiej wydajności.
    Może przetwarzać 1000+ pacjentów w <100ms.

    Returns:
        Predykcje, poziomy ryzyka i opcjonalne czynniki ryzyka dla każdego pacjenta.
    """
    start_time = time.perf_counter()

    results = []
    errors = []
    mode = app_state.get_current_mode()

    # Check if API mode is required but model not loaded
    if mode == "unavailable":
        raise HTTPException(
            status_code=503,
            detail="Model nie jest załadowany i tryb demo jest wyłączony. Ustaw ALLOW_DEMO=true lub załaduj model."
        )

    n_patients = len(request.patients)

    if app_state.is_loaded and app_state.model is not None:
        # PRODUCTION MODE: Vectorized batch prediction
        try:
            # Convert all patients to matrix at once
            X = patients_to_matrix(request.patients, app_state.feature_names)

            # Single vectorized prediction call
            probabilities, predictions = batch_predict_vectorized(X, app_state.model)

            # Extract risk factors if requested
            if request.include_risk_factors:
                risk_factors = get_batch_risk_factors(
                    X, app_state.feature_names, request.top_n_factors
                )
            else:
                risk_factors = [None] * n_patients

            # Build results
            for i in range(n_patients):
                prob = float(probabilities[i])
                results.append(BatchPatientResult(
                    index=i,
                    prediction=PredictionOutput(
                        probability=prob,
                        risk_level=get_risk_level_from_probability(prob),
                        prediction=int(predictions[i])
                    ),
                    top_risk_factors=risk_factors[i],
                    processing_status="success"
                ))

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Fallback to individual processing
            for i, patient in enumerate(request.patients):
                try:
                    pred = await predict(patient)
                    results.append(BatchPatientResult(
                        index=i,
                        prediction=pred,
                        processing_status="success"
                    ))
                except Exception as inner_e:
                    errors.append(BatchProcessingError(
                        patient_index=i,
                        error_type="prediction",
                        error_message=str(inner_e),
                        is_recoverable=True
                    ))
    else:
        # DEMO MODE: Use heuristic predictions
        for i, patient in enumerate(request.patients):
            demo_pred = get_demo_prediction(patient)
            demo_factors = None

            if request.include_risk_factors:
                demo_exp = get_demo_explanation(patient)
                demo_factors = [
                    RiskFactorItem(
                        feature=f["feature"],
                        value=f["value"],
                        importance=abs(f["contribution"]),
                        direction="increases_risk" if f["contribution"] > 0 else "decreases_risk"
                    )
                    for f in (demo_exp["risk_factors"] + demo_exp["protective_factors"])[:request.top_n_factors]
                ]

            results.append(BatchPatientResult(
                index=i,
                prediction=demo_pred,
                top_risk_factors=demo_factors,
                processing_status="demo"
            ))

    # Calculate summary statistics
    probs = [r.prediction.probability for r in results if r.processing_status != "error"]

    summary = BatchSummary(
        total_count=n_patients,
        low_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.LOW),
        moderate_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.MODERATE),
        high_risk_count=sum(1 for r in results if r.prediction.risk_level == RiskLevel.HIGH),
        avg_probability=float(np.mean(probs)) if probs else 0.0,
        median_probability=float(np.median(probs)) if probs else 0.0,
        min_probability=float(np.min(probs)) if probs else 0.0,
        max_probability=float(np.max(probs)) if probs else 0.0
    )

    processing_time = (time.perf_counter() - start_time) * 1000

    return BatchPredictionOutput(
        total_patients=n_patients,
        processed_count=len(results),
        success_count=len([r for r in results if r.processing_status in ["success", "demo"]]),
        error_count=len(errors),
        processing_time_ms=processing_time,
        mode=mode,
        summary=summary,
        results=results,
        errors=errors
    )


# ============================================================================
# CONFIG ENDPOINTS
# ============================================================================

@app.get("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
async def get_demo_mode_status():
    """
    Pobierz status konfiguracji trybu demo.

    Returns:
        Czy tryb demo jest dozwolony, status modelu i aktualny tryb pracy.
    """
    return DemoModeStatus(
        demo_allowed=app_state.allow_demo,
        model_loaded=app_state.is_loaded,
        current_mode=app_state.get_current_mode(),
        force_api_mode=app_state.force_api_mode
    )


@app.post("/config/demo-mode", response_model=DemoModeStatus, tags=["Config"])
async def set_demo_mode(request: DemoModeRequest):
    """
    Włącz lub wyłącz tryb demo.

    Gdy tryb demo jest wyłączony i model nie jest załadowany, API zwróci błędy 503.
    Nie można włączyć trybu demo gdy zmienna FORCE_API_MODE jest ustawiona.
    """
    try:
        app_state.set_demo_mode(request.enabled)
        return await get_demo_mode_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain/shap", response_model=SHAPExplanation, tags=["XAI"])
async def explain_shap(request: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie SHAP.

    Zwraca wartości SHAP dla cech pacjenta. Obsługuje XGBoost, Random Forest i LightGBM
    (parametr model_key).
    """
    try:
        model = _load_model_by_key(request.model_key)
        model_label = _MODEL_LABELS.get(request.model_key or "xgboost", "XGBoost")

        prediction = await predict(request.patient)
        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)

        # Użyj preinit SHAP explainer dla XGBoost, stwórz nowy dla innych modeli
        if (request.model_key in ("xgboost", None)) and app_state.shap_explainer is not None:
            real_exp = app_state.shap_explainer.explain_instance(instance)
        else:
            from src.xai.shap_explainer import SHAPExplainer
            X_bg = joblib.load(Path("models/saved/X_background.joblib"))
            explainer = SHAPExplainer(
                model=model,
                X_background=X_bg,
                feature_names=app_state.feature_names,
            )
            real_exp = explainer.explain_instance(instance)

        shap_values = {}
        contributions = []
        risk_factors = []
        protective_factors = []

        for fi in real_exp['feature_impacts'][:request.num_features]:
            shap_values[fi['feature']] = fi['shap_value']
            direction = "increases_risk" if fi['shap_value'] > 0 else "decreases_risk"
            entry = {
                "feature": fi['feature'],
                "value": fi['feature_value'],
                "contribution": fi['shap_value'],
                "direction": direction
            }
            contributions.append(entry)
            if fi['shap_value'] > 0:
                risk_factors.append(entry)
            elif fi['shap_value'] < 0:
                protective_factors.append(entry)

        return SHAPExplanation(
            base_value=real_exp['base_value'],
            shap_values=shap_values,
            feature_contributions=contributions,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            prediction=prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd SHAP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/lime", response_model=LIMEExplanation, tags=["XAI"])
async def explain_lime(request: ExplanationRequest):
    """
    Wygeneruj wyjaśnienie LIME.

    Zwraca wagi cech z lokalnego modelu zastępczego. Obsługuje XGBoost, Random Forest i LightGBM.
    """
    try:
        model = _load_model_by_key(request.model_key)
        prediction = await predict(request.patient)

        # Użyj preinit LIME explainer dla XGBoost, stwórz nowy dla innych modeli
        if (request.model_key in ("xgboost", None)) and app_state.lime_explainer is not None:
            lime_exp = app_state.lime_explainer
        else:
            from src.xai.lime_explainer import LIMEExplainer
            X_train = joblib.load(Path("models/saved/X_train.joblib"))
            lime_exp = LIMEExplainer(
                model=model,
                X_train=X_train,
                feature_names=app_state.feature_names,
            )

        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)
        real_exp = lime_exp.explain_instance(instance, num_features=request.num_features)

        feature_weights = []
        risk_factors = []
        protective_factors = []

        for feat_desc, weight in real_exp['feature_weights'][:request.num_features]:
            feature_weights.append({
                "feature": feat_desc,
                "weight": weight,
                "condition": feat_desc
            })
            entry = {"feature": feat_desc, "weight": weight}
            if weight > 0:
                risk_factors.append(entry)
            else:
                protective_factors.append(entry)

        return LIMEExplanation(
            intercept=real_exp['intercept'],
            feature_weights=feature_weights,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            local_prediction=real_exp['local_prediction'],
            prediction=prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd LIME: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/comparison", response_model=ComparisonResult, tags=["XAI"])
async def explain_comparison(request: ExplanationRequest):
    """
    Porównaj wyjaśnienia SHAP i LIME dla pacjenta.

    Zwraca rankingi cech z obu metod oraz miary zgodności.
    Obsługuje XGBoost, Random Forest i LightGBM.
    """
    try:
        model = _load_model_by_key(request.model_key)
        features = patient_to_array(request.patient, app_state.feature_names)
        instance = np.array(features)
        num_features = request.num_features

        shap_ranking: List[str] = []
        lime_ranking: List[str] = []

        # SHAP ranking
        if (request.model_key in ("xgboost", None)) and app_state.shap_explainer is not None:
            shap_exp = app_state.shap_explainer.explain_instance(instance)
        else:
            from src.xai.shap_explainer import SHAPExplainer
            X_bg = joblib.load(Path("models/saved/X_background.joblib"))
            shap_expl = SHAPExplainer(model=model, X_background=X_bg, feature_names=app_state.feature_names)
            shap_exp = shap_expl.explain_instance(instance)
        shap_ranking = [fi['feature'] for fi in shap_exp['feature_impacts'][:num_features]]

        # LIME ranking
        if (request.model_key in ("xgboost", None)) and app_state.lime_explainer is not None:
            lime_exp_obj = app_state.lime_explainer
        else:
            from src.xai.lime_explainer import LIMEExplainer
            X_train = joblib.load(Path("models/saved/X_train.joblib"))
            lime_exp_obj = LIMEExplainer(model=model, X_train=X_train, feature_names=app_state.feature_names)
        lime_result = lime_exp_obj.explain_instance(instance, num_features=num_features)
        for feat_desc, _ in lime_result['feature_weights'][:num_features]:
            for fn in app_state.feature_names:
                if fn in feat_desc:
                    if fn not in lime_ranking:
                        lime_ranking.append(fn)
                    break

        # Oblicz zgodność
        shap_set = set(shap_ranking[:num_features])
        lime_set = set(lime_ranking[:num_features])
        common = list(shap_set & lime_set)
        union = shap_set | lime_set
        agreement = len(common) / len(union) if union else 0.0

        # Korelacja Spearmana (na wspólnych cechach)
        spearman_corr = 0.0
        if len(common) >= 2:
            from scipy.stats import spearmanr
            shap_ranks = [shap_ranking.index(f) for f in common if f in shap_ranking]
            lime_ranks = [lime_ranking.index(f) for f in common if f in lime_ranking]
            if len(shap_ranks) >= 2:
                corr, _ = spearmanr(shap_ranks, lime_ranks)
                spearman_corr = float(corr) if not np.isnan(corr) else 0.0

        return ComparisonResult(
            methods_compared=["SHAP", "LIME"],
            ranking_agreement=agreement,
            common_top_features=common,
            individual_rankings={"SHAP": shap_ranking, "LIME": lime_ranking},
            spearman_correlations={"SHAP_vs_LIME": spearman_corr}
        )

    except Exception as e:
        logger.error(f"Błąd comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DALEX EXPLANATION
# ============================================================================

@app.post("/explain/dalex", response_model=DALEXExplanation, tags=["XAI"])
async def explain_dalex(request: ExplanationRequest):
    """
    Wyjaśnienie DALEX Break Down + Permutation Importance.

    Zwraca break-down analysis (lokalne wyjaśnienie) oraz globalną ważność cech
    za pomocą permutacji. Obsługuje XGBoost, Random Forest i LightGBM.
    """
    try:
        model = _load_model_by_key(request.model_key)
        model_label = _MODEL_LABELS.get(request.model_key or "xgboost", "XGBoost")

        from src.xai.dalex_wrapper import DALEXWrapper

        features = patient_to_array(request.patient, app_state.feature_names)
        X = np.array(features).reshape(1, -1)

        # Background data for explainer
        X_bg = joblib.load(Path("models/saved/X_background.joblib"))
        y_bg = joblib.load(Path("models/saved/y_background.joblib"))

        wrapper = DALEXWrapper(
            model=model,
            X=X_bg,
            y=y_bg,
            feature_names=app_state.feature_names,
            label=model_label
        )

        # Break-down for this patient
        bd = wrapper.explain_instance_break_down(X[0])

        # Variable importance (global)
        try:
            vi = wrapper.get_variable_importance(B=5)
        except Exception:
            vi = None

        risk_factors = [
            {"feature": c["variable"], "contribution": c["contribution"]}
            for c in bd.get("risk_factors", [])
        ]
        protective_factors = [
            {"feature": c["variable"], "contribution": c["contribution"]}
            for c in bd.get("protective_factors", [])
        ]

        return DALEXExplanation(
            intercept=bd["intercept"],
            prediction=bd["prediction"],
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            variable_importance=vi,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd DALEX: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EBM EXPLANATION
# ============================================================================

# EBM is trained once on startup (lazy init)
_ebm_model = None
_ebm_lock = False


def _ensure_ebm():
    """Train EBM model on the training data (lazy, once)."""
    global _ebm_model, _ebm_lock
    if _ebm_model is not None:
        return _ebm_model
    if _ebm_lock:
        return None

    _ebm_lock = True
    try:
        from src.xai.ebm_explainer import EBMExplainer

        X_train = joblib.load(Path("models/saved/X_train.joblib"))
        y_train = joblib.load(Path("models/saved/y_train.joblib"))
        with open("models/saved/feature_names.json") as f:
            feature_names = json.load(f)

        ebm = EBMExplainer(
            feature_names=feature_names,
            max_rounds=5000,
            interactions=10,
            random_state=42,
        )
        X_df = pd.DataFrame(X_train, columns=feature_names)
        ebm.fit(X_df, y_train, feature_names=feature_names)
        _ebm_model = ebm
        logger.info("EBM model wytrenowany")
        return ebm
    except Exception as e:
        logger.warning(f"Nie udało się wytrenować EBM: {e}")
        return None
    finally:
        _ebm_lock = False


@app.post("/explain/ebm", response_model=EBMExplanation, tags=["XAI"])
async def explain_ebm(request: ExplanationRequest):
    """
    Wyjaśnienie EBM (Explainable Boosting Machine).

    Zwraca lokalne wyjaśnienie + globalną ważność cech z inherentnie
    interpretowalnego modelu GAM.
    """
    try:
        ebm = _ensure_ebm()
        if ebm is None:
            raise HTTPException(status_code=503, detail="Model EBM niedostępny")

        features = patient_to_array(request.patient, app_state.feature_names)
        X = np.array(features).reshape(1, -1)

        # Local explanation
        local = ebm.explain_local(X[0])

        # Global importance
        global_imp = ebm.get_feature_importance()

        prob = local["probability_positive"]
        pred = local["prediction"]

        contributions = [
            {"feature": c["feature"], "contribution": c["score"]}
            for c in local.get("contributions", [])
        ]

        # Get top interactions
        global_data = ebm.explain_global()
        interactions = global_data.get("interactions_detected", [])

        return EBMExplanation(
            prediction=pred,
            probability=prob,
            risk_level=get_risk_level_from_probability(prob),
            global_importance=global_imp,
            local_contributions=contributions,
            interactions=interactions[:10],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd EBM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/patient", response_model=PatientExplanation, tags=["XAI"])
async def explain_for_patient(request: PatientExplanationRequest):
    """
    Wygeneruj wyjaśnienie zrozumiałe dla pacjenta.

    Dostosowuje język i poziom szczegółowości do poziomu health literacy.
    """
    try:
        prediction = await predict(request.patient)
        demo_exp = get_demo_explanation(request.patient)

        # Tłumaczenia cech
        translations = {
            "Wiek": "Twój wiek",
            "Manifestacja_Nerki": "Stan nerek",
            "Manifestacja_Sercowo_Naczyniowy": "Stan układu krążenia",
            "Zaostrz_Wymagajace_OIT": "Przebyte poważne zaostrzenia",
            "Liczba_Zajetych_Narzadow": "Liczba dotkniętych narządów",
            "Kreatynina": "Wskaźnik czynności nerek",
            "Max_CRP": "Poziom stanu zapalnego"
        }

        # Poziom ryzyka
        if prediction.probability < 0.3:
            risk_desc = "Analiza wskazuje na niskie ryzyko. To dobra wiadomość!"
        elif prediction.probability < 0.7:
            risk_desc = "Analiza wskazuje na umiarkowane ryzyko. Warto zwrócić uwagę na kilka czynników."
        else:
            risk_desc = "Analiza wskazuje na podwyższone ryzyko. Ważna jest regularna opieka lekarska."

        main_concerns = [
            translations.get(rf["feature"], rf["feature"])
            for rf in demo_exp["risk_factors"][:3]
        ]

        positive_factors = [
            translations.get(pf["feature"], pf["feature"])
            for pf in demo_exp["protective_factors"][:3]
        ]

        return PatientExplanation(
            risk_level=prediction.risk_level.value,
            risk_description=risk_desc,
            main_concerns=main_concerns if main_concerns else ["Brak szczególnych czynników ryzyka"],
            positive_factors=positive_factors if positive_factors else ["Analiza trwa"],
            recommendations="Zalecamy omówienie tych wyników z lekarzem prowadzącym.",
            technical_summary={
                "probability": prediction.probability,
                "n_risk_factors": len(demo_exp["risk_factors"]),
                "n_protective_factors": len(demo_exp["protective_factors"])
            } if request.health_literacy != HealthLiteracyLevel.BASIC else None,
            disclaimer="To narzędzie ma charakter informacyjny i nie zastępuje porady lekarza."
        )

    except Exception as e:
        logger.error(f"Błąd wyjaśnienia pacjenta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/global-importance", response_model=GlobalImportance, tags=["Model"])
async def get_global_importance():
    """
    Pobierz globalną ważność cech.

    Zwraca ranking cech według ich wpływu na predykcje modelu.
    """
    # Demo importance
    importance = {
        "Wiek": 0.15,
        "Manifestacja_Nerki": 0.12,
        "Zaostrz_Wymagajace_OIT": 0.11,
        "Liczba_Zajetych_Narzadow": 0.10,
        "Manifestacja_Sercowo-Naczyniowy": 0.09,
        "Kreatynina": 0.08,
        "Max_CRP": 0.07,
        "Dializa": 0.06,
        "Manifestacja_Zajecie_CSN": 0.05,
        "Plazmaferezy": 0.04
    }

    return GlobalImportance(
        feature_importance=importance,
        top_features=list(importance.keys()),
        method="SHAP TreeExplainer (demo)",
        n_samples=100
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Pobierz informacje o modelu.

    Zwraca metadane modelu i metryki wydajności.
    """
    return ModelInfo(
        model_type="XGBoostClassifier" if app_state.is_loaded else "Demo Model",
        n_features=len(app_state.feature_names) if app_state.feature_names else 20,
        feature_names=app_state.feature_names or ["Wiek", "Plec", "Manifestacja_Nerki", "..."],
        training_date="2024-01-15" if app_state.is_loaded else None,
        performance_metrics={
            "auc_roc": 0.85,
            "sensitivity": 0.82,
            "specificity": 0.78,
            "ppv": 0.65,
            "npv": 0.90
        },
        version="1.0.0"
    )


# ============================================================================
# AGENT PREDICTION TOOL
# ============================================================================

async def run_prediction_tool(patient: PatientInput) -> AgentPredictionData:
    """
    Wykonaj predykcję z wyjaśnieniem SHAP jako narzędzie agenta.

    Zwraca dane predykcji + czynniki SHAP gotowe do wyświetlenia w czacie.
    """
    # 1. Predykcja
    pred = await predict(patient)

    factors: List[AgentPredictionFactor] = []
    base_value = 0.0

    # 2. SHAP explanation (jeśli dostępny)
    try:
        if app_state.shap_explainer is not None and app_state.feature_names:
            features = patient_to_array(patient, app_state.feature_names)
            instance = np.array(features)
            real_exp = app_state.shap_explainer.explain_instance(instance)
            base_value = real_exp.get('base_value', 0.0)

            for fi in real_exp['feature_impacts'][:10]:
                direction = "increases_risk" if fi['shap_value'] > 0 else "decreases_risk"
                factors.append(AgentPredictionFactor(
                    feature=fi['feature'],
                    contribution=fi['shap_value'],
                    direction=direction,
                ))
        else:
            # Fallback to demo explanation factors
            demo_exp = get_demo_explanation(patient)
            base_value = demo_exp.get('base_value', 0.0)
            for rf in demo_exp["risk_factors"]:
                factors.append(AgentPredictionFactor(
                    feature=rf["feature"],
                    contribution=rf["contribution"],
                    direction=rf.get("direction", "increases_risk"),
                ))
            for pf in demo_exp["protective_factors"]:
                factors.append(AgentPredictionFactor(
                    feature=pf["feature"],
                    contribution=pf["contribution"],
                    direction=pf.get("direction", "decreases_risk"),
                ))
    except Exception as e:
        logger.warning(f"SHAP explanation failed in agent tool: {e}")

    return AgentPredictionData(
        prediction=pred,
        factors=factors,
        base_value=base_value,
    )


@app.post("/agent/predict", response_model=AgentPredictionData, tags=["Agent"])
async def agent_predict(patient: PatientInput):
    """
    Narzędzie predykcji agenta — endpoint do wywoływania przez agenta.

    Wykonuje predykcję ryzyka śmiertelności i zwraca dane z wyjaśnieniem SHAP,
    gotowe do renderowania w interfejsie czatu (wykresy, czynniki).
    """
    try:
        return await run_prediction_tool(patient)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd agent predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AGENT CONVERSATION (conversational data collection)
# ============================================================================

# Kolekcja pytań do zebrania danych pacjenta
_COLLECTION_STEPS = [
    {
        "field": "wiek_rozpoznania",
        "question": "Witaj! Jestem asystentem medycznym systemu Vasculitis XAI.\n\nPomogę Ci ocenić ryzyko śmiertelności u pacjenta z zapaleniem naczyń. Zaczniemy od kilku pytań.\n\n**Jaki jest wiek pacjenta w momencie rozpoznania choroby?** (podaj liczbę lat)",
        "type": "number",
        "default": 50,
        "widget": "slider",
        "min": 0,
        "max": 100,
        "step": 1,
        "unit": "lat",
    },
    {
        "field": "opoznienie_rozpoznia",
        "question": "Ile miesięcy minęło od pierwszych objawów do postawienia diagnozy?\n(Jeśli nie wiesz, wpisz 0)",
        "type": "number",
        "default": 0,
        "widget": "slider",
        "min": 0,
        "max": 24,
        "step": 1,
        "unit": "mies.",
    },
    {
        "field": "manifestacja_nerki",
        "question": "Czy choroba dotknęła **nerek**? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "manifestacja_sercowo_naczyniowy",
        "question": "Czy występują objawy ze strony **układu sercowo-naczyniowego**? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "manifestacja_zajecie_csn",
        "question": "Czy choroba dotknęła **ośrodkowy układ nerwowy** (mózg, rdzeń)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "manifestacja_neurologiczny",
        "question": "Czy występują objawy **neurologiczne obwodowe** (neuropatia)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
    {
        "field": "liczba_zajetych_narzadow",
        "question": "Ile **narządów ogólnie** jest objętych chorobą? (podaj liczbę 1-10)",
        "type": "number",
        "default": 1,
        "widget": "slider",
        "min": 0,
        "max": 5,
        "step": 1,
        "unit": "",
    },
    {
        "field": "zaostrz_wymagajace_hospital",
        "question": "Czy wystąpiły zaostrzenia wymagające **hospitalizacji**? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie"],
    },
    {
        "field": "zaostrz_wymagajace_oit",
        "question": "Czy wystąpiły zaostrzenia wymagające pobytu na **OIT** (oddział intensywnej terapii)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie"],
    },
    {
        "field": "kreatynina",
        "question": "Jaki jest poziom **kreatyniny** (μmol/L)?\n(Norma: 60-110. Jeśli nieznane, wpisz 0)",
        "type": "number",
        "default": 0,
        "widget": "slider",
        "min": 80,
        "max": 300,
        "step": 1,
        "unit": "μmol/L",
    },
    {
        "field": "czas_sterydow",
        "question": "Ile **miesięcy** pacjent jest leczony sterydami? (podaj liczbę lub 0)",
        "type": "number",
        "default": 0,
        "widget": "slider",
        "min": 0,
        "max": 36,
        "step": 1,
        "unit": "mies.",
    },
    {
        "field": "plazmaferezy",
        "question": "Czy pacjentowi wykonywano **plazmaferezy** (oczyszczanie krwi)? (tak/nie)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie"],
    },
    {
        "field": "biopsja_wynik",
        "question": "Czy wynik **biopsji** był dodatni? (tak/nie/nie wiem)",
        "type": "boolean",
        "default": 0,
        "widget": "buttons",
        "options": ["tak", "nie", "nie wiem"],
    },
]

# Domyślne wartości dla pól, które nie są zbierane przez agenta
_DEFAULT_EXTRA_FIELDS = {
    "manifestacja_miesno_szkiel": 0,
    "manifestacja_skora": 0,
    "manifestacja_wzrok": 0,
    "manifestacja_pokarmowy": 0,
    "manifestacja_moczowo_plciowy": 0,
    "eozynofilia_krwi_obwodowej_wartosc": 0,
    "pulsy": 0,
}


def _parse_boolean(value: str) -> int:
    """Parsuj odpowiedź tak/nie na int 0/1."""
    v = value.strip().lower()
    if v in ('tak', 'yes', 't', 'y', '1'):
        return 1
    if v in ('nie', 'no', 'n', '0'):
        return 0
    if 'tak' in v:
        return 1
    if 'nie' in v:
        return 0
    return 0


def _parse_number(value: str) -> Optional[float]:
    """Parsuj odpowiedź liczbową."""
    import re
    match = re.search(r'[\d]+[.,]?[\d]*', value.replace(',', '.'))
    if match:
        return float(match.group().replace(',', '.'))
    return None


def _is_skip(value: str) -> bool:
    """Sprawdź czy użytkownik chce pominąć pytanie."""
    v = value.strip().lower()
    return v in ('nie wiem', 'pomin', 'skip', 'nie znam', '-', 'n/a', 'brak')


@app.post("/agent/chat", response_model=AgentConversationResponse, tags=["Agent"])
async def agent_conversation(request: AgentConversationRequest):
    """
    Konwersacyjny agent zbierający dane pacjenta i wykonujący predykcję.

    Flow:
    1. Agent zadaje pytania jedno po drugim
    2. Zbiera odpowiedzi i buduje profil pacjenta
    3. Gdy wszystkie dane zebrane → wykonuje predykcję
    4. Pokazuje wyniki z wykresami
    5. Pozwala na dalszą dyskusję o wynikach i chorobie
    """
    try:
        collected = dict(request.collected_data)
        current_step = request.current_step
        phase = request.phase
        user_message = request.message.strip()

        # ========================================
        # Phase: DISCUSSION (after prediction)
        # ========================================
        if phase == "discussion":
            message_lower = user_message.lower()

            if any(kw in message_lower for kw in ['nowy', 'nowa', 'restart', 'jeszcze raz', 'od nowa']):
                return AgentConversationResponse(
                    response="Okej, zaczynamy od nowa!\n\n" + _COLLECTION_STEPS[0]["question"],
                    collected_data={},
                    current_step=1,
                    phase="collecting",
                    missing_fields=[s["field"] for s in _COLLECTION_STEPS],
                    follow_up_suggestions=_COLLECTION_STEPS[0].get("suggestions", ["40", "55", "65", "75"]),
                )

            if any(kw in message_lower for kw in ['czynnik', 'wpływa', 'dlaczego', 'shap']):
                patient_data = dict(collected)
                patient_data.update(_DEFAULT_EXTRA_FIELDS)
                pred_data = await run_prediction_tool(PatientInput(**patient_data))
                factor_text = "Szczegółowe zestawienie czynników:\n\n"
                sorted_f = sorted(pred_data.factors, key=lambda f: abs(f.contribution), reverse=True)
                for i, f in enumerate(sorted_f[:8], 1):
                    direction = "↑ zwiększa" if f.contribution > 0 else "↓ zmniejsza"
                    factor_text += f"{i}. **{f.feature}**: {direction} ryzyko ({f.contribution:+.3f})\n"
                factor_text += "\nCzy chcesz dowiedzieć się więcej o konkretnym czynniku?"
                return AgentConversationResponse(
                    response=factor_text,
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    prediction_data=pred_data,
                    follow_up_suggestions=["Co mogę zrobić?", "Opowiedz o zapaleniu naczyń", "Nowy pacjent"],
                )

            if any(kw in message_lower for kw in ['zalec', 'co robić', 'co robic', 'pomoc', 'porad']):
                return AgentConversationResponse(
                    response="""Oto moje zalecenia:

1. **Regularne konsultacje** z lekarzem prowadzącym
2. **Przestrzeganie zaleceń** dotyczących leczenia
3. **Zgłaszanie objawów** — informuj lekarza o zmianach
4. **Badania kontrolne** — regularnie wykonuj zalecone badania

Pamiętaj, że ostateczne decyzje dotyczące leczenia podejmuje lekarz prowadzący.

Czy masz jeszcze pytania?""",
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Opowiedz o zapaleniu naczyń", "Nowy pacjent"],
                )

            if any(kw in message_lower for kw in ['zapalenie naczyń', 'vasculitis', 'chorob', 'czego']):
                return AgentConversationResponse(
                    response="""**Zapalenie naczyń** (vasculitis) to grupa chorób, w których dochodzi do zapalenia ścian naczyń krwionośnych.

**Główne typy:**
- GPA (ziarniniakowatość z zapaleniem naczyń)
- MPA (mikroskopowe zapalenie naczyń)
- EGPA (eozynofilowa ziarniniakowatość, Churg-Strauss)

**Czynniki prognostyczne:**
- Wiek w momencie rozpoznania
- Zajęcie nerek, serca, OUN
- Liczba zajętych narządów
- Przebieg zaostrzeń

**Leczenie:** glikokortykosteroidy, cyklofosfamid, rytuksymab, a w ciężkich przypadkach plazmafereza.

Czy chcesz wiedzieć więcej?""",
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Co mogę zrobić?", "Nowy pacjent"],
                )

            # General discussion
            return AgentConversationResponse(
                response="""Mogę pomóc Ci z:

- **Czynniki ryzyka** -- co wpływa na wynik
- **Zalecenia** -- na co zwrócić uwagę
- **Zapalenie naczyń** -- informacje o chorobie
- **Nowy pacjent** -- rozpocznij od nowa

O co chciałbyś zapytać?""",
                collected_data=collected,
                current_step=current_step,
                phase="discussion",
                follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Opowiedz o zapaleniu naczyń", "Nowy pacjent"],
            )

        # ========================================
        # Phase: COLLECTING data
        # ========================================
        if phase == "collecting":
            # Parse the user's answer for the current step
            if current_step > 0 and current_step <= len(_COLLECTION_STEPS):
                step = _COLLECTION_STEPS[current_step - 1]

                if _is_skip(user_message):
                    collected[step["field"]] = step["default"]
                elif step["type"] == "boolean":
                    collected[step["field"]] = _parse_boolean(user_message)
                elif step["type"] == "number":
                    parsed = _parse_number(user_message)
                    collected[step["field"]] = parsed if parsed is not None else step["default"]
                else:
                    collected[step["field"]] = step["default"]

            # Check if we've collected all steps
            if current_step >= len(_COLLECTION_STEPS):
                # All data collected → prediction!
                patient_data = dict(collected)
                patient_data.update(_DEFAULT_EXTRA_FIELDS)

                pred_data = await run_prediction_tool(PatientInput(**patient_data))

                risk_level_pl = {
                    "low": "Niskie",
                    "moderate": "Umiarkowane",
                    "high": "Wysokie",
                }.get(pred_data.prediction.risk_level.value, pred_data.prediction.risk_level.value)

                risk_factors = [f for f in pred_data.factors if f.contribution > 0][:5]
                protective_factors = [f for f in pred_data.factors if f.contribution < 0][:5]

                response = f"""**Zebrałem wszystkie dane! Oto wynik analizy:**

**Poziom ryzyka:** {risk_level_pl}
**Prawdopodobieństwo:** {pred_data.prediction.probability:.1%}
"""

                if risk_factors:
                    response += "\n**Czynniki zwiększające ryzyko:**\n"
                    for f in risk_factors:
                        response += f"- {f.feature}: +{f.contribution:.3f}\n"
                if protective_factors:
                    response += "\n**Czynniki zmniejszające ryzyko:**\n"
                    for f in protective_factors:
                        response += f"- {f.feature}: {f.contribution:.3f}\n"

                response += """\nPoniżej znajdziesz interaktywne wykresy. Możesz mnie teraz pytać o:\n- Czynniki wpływające na wynik\n- Zalecenia\n- Informacje o zapaleniu naczyń\n- Rozpocząć analizę nowego pacjenta"""

                return AgentConversationResponse(
                    response=response,
                    collected_data=collected,
                    current_step=current_step,
                    phase="discussion",
                    prediction_data=pred_data,
                    follow_up_suggestions=["Jakie czynniki wpływają na wynik?", "Co mogę zrobić?", "Nowy pacjent"],
                )

            # Ask the next question
            next_step_idx = current_step
            if next_step_idx < len(_COLLECTION_STEPS):
                next_step = _COLLECTION_STEPS[next_step_idx]
                progress = f"[{next_step_idx + 1}/{len(_COLLECTION_STEPS)}]"
                question = f"{progress} {next_step['question']}"

                # Build field_meta for frontend widget rendering
                field_meta = {
                    "field": next_step["field"],
                    "type": next_step["type"],
                    "widget": next_step.get("widget", "input"),
                }
                if next_step.get("min") is not None:
                    field_meta["min"] = next_step["min"]
                if next_step.get("max") is not None:
                    field_meta["max"] = next_step["max"]
                if next_step.get("step") is not None:
                    field_meta["step"] = next_step["step"]
                if next_step.get("unit"):
                    field_meta["unit"] = next_step["unit"]
                if next_step.get("options"):
                    field_meta["options"] = next_step["options"]

                return AgentConversationResponse(
                    response=question,
                    collected_data=collected,
                    current_step=next_step_idx + 1,
                    phase="collecting",
                    missing_fields=[s["field"] for s in _COLLECTION_STEPS[next_step_idx:]],
                    follow_up_suggestions=next_step.get("options", []),
                    field_meta=field_meta,
                )

        # Fallback — start over
        return AgentConversationResponse(
            response="Zaczynamy od nowa!\n\n" + _COLLECTION_STEPS[0]["question"],
            collected_data={},
            current_step=1,
            phase="collecting",
            missing_fields=[s["field"] for s in _COLLECTION_STEPS],
            follow_up_suggestions=_COLLECTION_STEPS[0].get("options", []),
            field_meta={
                "field": _COLLECTION_STEPS[0]["field"],
                "type": _COLLECTION_STEPS[0]["type"],
                "widget": _COLLECTION_STEPS[0].get("widget", "input"),
                **({"min": _COLLECTION_STEPS[0]["min"], "max": _COLLECTION_STEPS[0]["max"], "step": _COLLECTION_STEPS[0]["step"], "unit": _COLLECTION_STEPS[0]["unit"]} if _COLLECTION_STEPS[0].get("widget") == "slider" else {}),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CHAT (z narzędziem predykcji)
# ============================================================================

# Intencje związane z predykcją
_PREDICTION_KEYWORDS = [
    'predykc', 'ryzyko', 'śmiertelno', 'smiertelno', 'prawdopodob',
    'wynik', 'analiz', 'ocen', 'jakie szanse', 'jakie ryzyko',
    'prognos', 'zgon', 'ryzyk', 'mortality', 'risk',
    'jakie jest ryzyko', 'oblicz', 'policz', 'sprawdź', 'sprawdz',
]

_FACTOR_KEYWORDS = [
    'czynnik', 'wpływa', 'dlaczego', 'powód', 'powod', 'przyczyn',
    'co zwiększa', 'co zmniejsza', 'shap', 'ważno', 'wplyw',
]

_RECOMMENDATION_KEYWORDS = [
    'pomoc', 'co robić', 'co robic', 'zalec', 'rekomend', 'sugesti',
    'porad', 'co mogę', 'co powinien', 'co powinienem',
]


def _build_prediction_response(pred_result: PredictionOutput, factor_data: list) -> str:
    """Zbuduj tekstową odpowiedź agenta z wynikami predykcji."""
    risk_level_pl = {
        "low": "Niskie",
        "moderate": "Umiarkowane",
        "high": "Wysokie",
    }.get(pred_result.risk_level.value, pred_result.risk_level.value)

    response = f"""Na podstawie wprowadzonych danych przeprowadziłem analizę ryzyka śmiertelności:

**Poziom ryzyka:** {risk_level_pl}
**Prawdopodobieństwo:** {pred_result.probability:.1%}
"""

    if factor_data:
        risk_factors = [f for f in factor_data if f.contribution > 0]
        protective_factors = [f for f in factor_data if f.contribution < 0]

        if risk_factors:
            response += "\n**Czynniki zwiększające ryzyko:**\n"
            for f in risk_factors[:5]:
                response += f"- {f.feature}: +{f.contribution:.3f}\n"

        if protective_factors:
            response += "\n**Czynniki zmniejszające ryzyko:**\n"
            for f in protective_factors[:5]:
                response += f"- {f.feature}: {f.contribution:.3f}\n"

    response += """\nPoniżej znajdziesz interaktywne wykresy z szczegółową analizą.\n\nCzy chciałbyś/chciałabyś dowiedzieć się więcej o konkretnych czynnikach?"""

    return response


def _build_factor_response(factor_data: list) -> str:
    """Zbuduj odpowiedź o czynnikach ryzyka."""
    if not factor_data:
        return "Brak danych o czynnikach ryzyka. Najpierw przeprowadź analizę pacjenta."

    sorted_factors = sorted(factor_data, key=lambda f: abs(f.contribution), reverse=True)

    response = "Oto szczegółowe zestawienie czynników wpływających na ocenę ryzyka:\n\n"

    for i, f in enumerate(sorted_factors[:8], 1):
        direction = "↑ zwiększa" if f.contribution > 0 else "↓ zmniejsza"
        response += f"{i}. **{f.feature}**: {direction} ryzyko (wpływ: {f.contribution:+.3f})\n"

    response += "\nCzy chcesz dowiedzieć się więcej o którymś z tych czynników?"
    return response


def _build_recommendation_response() -> str:
    """Zbuduj odpowiedź z zaleceniami."""
    return """Oto moje zalecenia:

1. **Regularne konsultacje** z lekarzem prowadzącym
2. **Przestrzeganie zaleceń** dotyczących leczenia
3. **Zgłaszanie objawów** — informuj lekarza o wszelkich niepokojących zmianach
4. **Monitorowanie** regularnie wykonuj zalecone badania kontrolne

Pamiętaj, że ta analiza jest narzędziem wspierającym — ostateczne decyzje
dotyczące leczenia powinny być podejmowane wspólnie z lekarzem.

Czy masz dodatkowe pytania?"""


def _build_general_response() -> str:
    """Zbuduj ogólną odpowiedź powitalną."""
    return """Jestem asystentem pomagającym zrozumieć wyniki analizy ryzyka śmiertelności w zapaleniu naczyń.

Mogę pomóc Ci z:
- **Predykcją ryzyka** — obliczę prawdopodobieństwo na podstawie danych pacjenta
- **Wyjaśnieniem czynników** — powiem co wpływa na wynik i dlaczego
- **Zaleceniami** — podpowiem na co zwrócić uwagę

Aby przeprowadzić analizę, potrzebuję danych pacjenta. Wypełnij formularz
po lewej stronie, a następnie zapytaj mnie o wynik.

O czym chciałbyś/chciałabyś porozmawiać?"""


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Rozmowa z agentem AI.

    Odpowiada na pytania pacjenta/klinicysty o wyniki analizy.
    Może wywołać narzędzie predykcji i zwrócić dane do wyświetlenia wykresów.
    """
    try:
        message = request.message.lower()
        prediction_data: Optional[AgentPredictionData] = None

        # Wykryj intencję — predykcja
        is_prediction_intent = any(kw in message for kw in _PREDICTION_KEYWORDS)
        is_factor_intent = any(kw in message for kw in _FACTOR_KEYWORDS)
        is_recommendation_intent = any(kw in message for kw in _RECOMMENDATION_KEYWORDS)

        if is_prediction_intent:
            # Użyj narzędzia predykcji
            pred_result = await run_prediction_tool(request.patient)
            prediction_data = pred_result
            response = _build_prediction_response(pred_result.prediction, pred_result.factors)

        elif is_factor_intent:
            # Najpierw pobierz predykcję, żeby mieć czynniki
            pred_result = await run_prediction_tool(request.patient)
            prediction_data = pred_result
            response = _build_factor_response(pred_result.factors)

        elif is_recommendation_intent:
            response = _build_recommendation_response()

        else:
            response = _build_general_response()

        # Dodaj disclaimer dla pacjentów
        if request.health_literacy != HealthLiteracyLevel.CLINICIAN:
            response += "\n\n---\n*Pamiętaj: to narzędzie informacyjne, nie zastępuje porady lekarza.*"

        return ChatResponse(
            response=response,
            detected_concerns=None,
            follow_up_suggestions=[
                "Jakie czynniki wpływają na wynik?",
                "Co mogę zrobić, aby poprawić swoje zdrowie?",
                "Czy powinienem/powinnam martwić się wynikiem?",
            ],
            prediction_data=prediction_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd chatu: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=exc.status_code
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Nieobsłużony błąd: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code=500
        ).dict()
    )


# ============================================================================
# MAIN
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Uruchom serwer API."""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True
    )


if __name__ == "__main__":
    run_server()
