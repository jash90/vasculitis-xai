"""
Testy jednostkowe dla API FastAPI.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schemas import (
    PatientInput, PredictionOutput, RiskLevel, HealthLiteracyLevel,
    patient_to_array, get_risk_level_from_probability
)


class TestSchemas:
    """Testy dla schematów Pydantic."""

    def test_patient_input_valid(self):
        """Test poprawnych danych pacjenta."""
        patient = PatientInput(
            wiek_rozpoznania=50,
            liczba_zajetych_narzadow=3,
            manifestacja_nerki=1,
            zaostrz_wymagajace_oit=0,
            kreatynina=100.0,
            eozynofilia_krwi_obwodowej_wartosc=0.5
        )

        assert patient.wiek_rozpoznania == 50
        assert patient.manifestacja_nerki == 1

    def test_patient_input_defaults(self):
        """Test wartości domyślnych."""
        patient = PatientInput(
            wiek_rozpoznania=50
        )

        assert patient.liczba_zajetych_narzadow == 0
        assert patient.manifestacja_nerki == 0
        assert patient.plazmaferezy == 0

    def test_patient_input_validation_age(self):
        """Test walidacji wieku."""
        with pytest.raises(ValueError):
            PatientInput(wiek_rozpoznania=-5)

        with pytest.raises(ValueError):
            PatientInput(wiek_rozpoznania=150)

    def test_patient_input_validation_binary_fields(self):
        """Test walidacji pól binarnych."""
        with pytest.raises(ValueError):
            PatientInput(wiek_rozpoznania=50, manifestacja_nerki=2)

    def test_prediction_output_valid(self):
        """Test poprawnego wyniku predykcji."""
        prediction = PredictionOutput(
            probability=0.35,
            risk_level=RiskLevel.MODERATE,
            prediction=0
        )

        assert prediction.probability == 0.35
        assert prediction.risk_level == RiskLevel.MODERATE

    def test_prediction_output_with_ci(self):
        """Test predykcji z przedziałem ufności."""
        prediction = PredictionOutput(
            probability=0.5,
            risk_level=RiskLevel.MODERATE,
            prediction=0,
            confidence_interval={"lower": 0.4, "upper": 0.6}
        )

        assert prediction.confidence_interval["lower"] == 0.4
        assert prediction.confidence_interval["upper"] == 0.6

    def test_get_risk_level_from_probability(self):
        """Test konwersji prawdopodobieństwa na poziom ryzyka."""
        assert get_risk_level_from_probability(0.1) == RiskLevel.LOW
        assert get_risk_level_from_probability(0.29) == RiskLevel.LOW
        assert get_risk_level_from_probability(0.3) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.5) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.69) == RiskLevel.MODERATE
        assert get_risk_level_from_probability(0.7) == RiskLevel.HIGH
        assert get_risk_level_from_probability(0.95) == RiskLevel.HIGH

    def test_patient_to_array(self):
        """Test konwersji pacjenta do tablicy."""
        patient = PatientInput(
            wiek_rozpoznania=55,
            manifestacja_nerki=1,
            kreatynina=100.0,
            plazmaferezy=1
        )

        feature_order = ['Wiek_rozpoznania', 'Manifestacja_Nerki', 'Kreatynina', 'Plazmaferezy']

        arr = patient_to_array(patient, feature_order)

        assert len(arr) == 4
        assert arr[0] == 55  # Wiek_rozpoznania
        assert arr[1] == 1   # Manifestacja_Nerki
        assert arr[2] == 100.0  # Kreatynina
        assert arr[3] == 1   # Plazmaferezy

    def test_health_literacy_levels(self):
        """Test poziomów health literacy."""
        assert HealthLiteracyLevel.BASIC.value == "basic"
        assert HealthLiteracyLevel.ADVANCED.value == "advanced"
        assert HealthLiteracyLevel.CLINICIAN.value == "clinician"

    def test_risk_level_enum(self):
        """Test enum poziomów ryzyka."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"


class TestAPIEndpoints:
    """Testy dla endpointów API (bez uruchamiania serwera)."""

    def test_import_api_main(self):
        """Test importu modułu API."""
        from src.api.main import app
        assert app is not None

    def test_api_routes_exist(self):
        """Test że wymagane ścieżki istnieją."""
        from src.api.main import app

        routes = [route.path for route in app.routes]

        assert "/" in routes
        assert "/health" in routes
        assert "/predict" in routes
        assert "/predict/auc" in routes
        assert "/models/auc" in routes
        assert "/explain/shap" in routes
        assert "/explain/lime" in routes
        assert "/explain/patient" in routes
        assert "/model/info" in routes
        assert "/chat" in routes

    def test_auc_raw_frame_preserves_training_order_and_missing_values(self):
        """Test budowania wejścia dla pełnego modelu AUC."""
        import numpy as np
        from src.api.main import _build_auc_raw_frame

        frame = _build_auc_raw_frame(
            {"B": "2.5", "A": None},
            ["A", "B", "C"],
        )

        assert frame.columns.tolist() == ["A", "B", "C"]
        assert np.isnan(frame.iloc[0]["A"])
        assert frame.iloc[0]["B"] == 2.5
        assert np.isnan(frame.iloc[0]["C"])


# Testy integracyjne (wymagają uruchomionego serwera)
@pytest.mark.skipif(True, reason="Wymaga uruchomionego serwera API")
class TestAPIIntegration:
    """Testy integracyjne API."""

    @pytest.fixture
    def client(self):
        """Fixture dla klienta testowego."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_predict(self, client):
        """Test predykcji."""
        patient_data = {
            "wiek_rozpoznania": 55,
            "liczba_zajetych_narzadow": 3,
            "manifestacja_nerki": 1
        }

        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "risk_level" in data

    def test_explain_shap(self, client):
        """Test wyjaśnienia SHAP."""
        request_data = {
            "patient": {
                "wiek_rozpoznania": 55,
                "manifestacja_nerki": 1
            },
            "method": "shap",
            "num_features": 5
        }

        response = client.post("/explain/shap", json=request_data)
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
