"""
Moduł SHAP Explainer dla wyjaśnień opartych na teorii gier.

Implementuje SHapley Additive exPlanations (SHAP)
dla modeli klasyfikacji ryzyka śmiertelności.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Klasa do generowania wyjaśnień SHAP.

    Zapewnia wyjaśnienia predykcji oparte na wartościach Shapleya,
    z optymalizacją TreeSHAP dla modeli drzewiastych.

    Attributes:
        model: Model do wyjaśniania
        feature_names: Lista nazw cech
        explainer: Obiekt SHAP Explainer
        explainer_type: Typ explainera (tree, kernel, linear)
    """

    def __init__(
        self,
        model,
        X_background: np.ndarray,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = 'auto'
    ):
        """
        Inicjalizacja SHAP Explainera.

        Args:
            model: Wytrenowany model
            X_background: Dane tła do obliczenia expected value
            feature_names: Lista nazw cech
            explainer_type: Typ explainera ('auto', 'tree', 'kernel', 'linear')
        """
        self.model = model
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_background.shape[1])]
        self.X_background = X_background

        # Wybierz odpowiedni explainer
        if explainer_type == 'auto':
            self.explainer_type = self._detect_explainer_type(model)
        else:
            self.explainer_type = explainer_type

        # Inicjalizuj explainer
        self.explainer = self._create_explainer(model, X_background)

        logger.info(f"SHAPExplainer ({self.explainer_type}) zainicjalizowany dla {len(self.feature_names)} cech")

    def _detect_explainer_type(self, model) -> str:
        """Wykryj optymalny typ explainera dla modelu."""
        model_type = type(model).__name__.lower()

        tree_models = [
            'randomforest', 'xgb', 'lgbm', 'lightgbm', 'gradientboosting',
            'decisiontree', 'extratrees', 'catboost'
        ]

        linear_models = ['logisticregression', 'linearregression', 'ridge', 'lasso']

        for tm in tree_models:
            if tm in model_type:
                return 'tree'

        for lm in linear_models:
            if lm in model_type:
                return 'linear'

        return 'kernel'

    def _create_explainer(self, model, X_background: np.ndarray):
        """Utwórz odpowiedni explainer."""
        if self.explainer_type == 'tree':
            try:
                return shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"TreeExplainer nie działa: {e}. Używam KernelExplainer.")
                self.explainer_type = 'kernel'

        if self.explainer_type == 'linear':
            try:
                return shap.LinearExplainer(model, X_background)
            except Exception as e:
                logger.warning(f"LinearExplainer nie działa: {e}. Używam KernelExplainer.")
                self.explainer_type = 'kernel'

        # Fallback to KernelExplainer
        # Użyj próbki tła dla szybkości
        if len(X_background) > 100:
            background_sample = shap.sample(X_background, 100)
        else:
            background_sample = X_background

        return shap.KernelExplainer(
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            background_sample
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        check_additivity: bool = False
    ) -> Dict[str, Any]:
        """
        Oblicz wartości SHAP dla pojedynczej instancji.

        Args:
            instance: Wektor cech (1D lub 2D array)
            check_additivity: Czy sprawdzić addytywność

        Returns:
            Słownik z wyjaśnieniem
        """
        # Upewnij się że instance jest 2D
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Oblicz wartości SHAP
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(instance, check_additivity=check_additivity)
        else:
            shap_values = self.explainer.shap_values(instance)

        # Dla klasyfikacji binarnej - weź wartości dla klasy pozytywnej
        # XGBoost zwraca (n_samples, n_features) = (1, 20)
        # Random Forest zwraca (n_samples, n_features, n_classes) = (1, 20, 2)
        # LightGBM zwraca list[ndarray(n_samples, n_features)]
        if isinstance(shap_values, list):
            # LightGBM: [class0_vals, class1_vals], each (1, n_features)
            shap_vals = shap_values[1][0]  # Klasa 1, pierwsza instancja
            base_value = self.explainer.expected_value[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Random Forest: (1, n_features, n_classes)
            shap_vals = shap_values[0, :, 1]  # klasa 1, spłaszczone do (n_features,)
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray) and base_value.ndim >= 1:
                base_value = float(base_value[1])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            # XGBoost: (n_samples, n_features) = (1, 20)
            shap_vals = shap_values[0]  # pierwsza (jedyna) instancja
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 1:
            # Rzadki przypadek: 1D shap values
            shap_vals = shap_values
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        else:
            shap_vals = np.array(shap_values).flatten()
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])

        # Predykcja
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance)[0]
            prediction = int(np.argmax(proba))
            probability_positive = float(proba[1])
        else:
            pred = self.model.predict(instance)[0]
            prediction = int(pred > 0.5)
            probability_positive = float(pred)

        # Przygotuj wyjaśnienie
        # instance może być 1D (20,) lub 2D (1,20) — spłaszcz do 1D
        instance_flat = instance.flatten() if isinstance(instance, np.ndarray) else np.array(instance).flatten()
        feature_impacts = list(zip(self.feature_names, shap_vals.tolist(), instance_flat.tolist()))
        feature_impacts_sorted = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)

        explanation = {
            'shap_values': shap_vals.tolist(),
            'base_value': float(base_value),
            'feature_values': instance_flat.tolist(),
            'feature_names': self.feature_names,
            'prediction': prediction,
            'probability_positive': probability_positive,
            'feature_impacts': [
                {
                    'feature': feat,
                    'shap_value': float(shap_val),
                    'feature_value': float(feat_val),
                    'direction': 'zwiększa ryzyko' if shap_val > 0 else 'zmniejsza ryzyko'
                }
                for feat, shap_val, feat_val in feature_impacts_sorted
            ],
            'explainer_type': self.explainer_type
        }

        # Podział na czynniki ryzyka i ochronne
        explanation['risk_factors'] = [
            fi for fi in explanation['feature_impacts'] if fi['shap_value'] > 0
        ]
        explanation['protective_factors'] = [
            fi for fi in explanation['feature_impacts'] if fi['shap_value'] < 0
        ]

        return explanation

    def explain_dataset(
        self,
        X: np.ndarray,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Oblicz wartości SHAP dla całego zbioru danych.

        Args:
            X: Macierz cech
            max_samples: Maksymalna liczba próbek

        Returns:
            Słownik z wynikami
        """
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        logger.info(f"Obliczanie SHAP dla {len(X_sample)} próbek...")

        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_sample)
        else:
            shap_values = self.explainer.shap_values(X_sample)

        # Dla klasyfikacji binarnej
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[1])
            base_value = self.explainer.expected_value[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Random Forest batch: (n_samples, n_features, n_classes)
            shap_vals = shap_values[:, :, 1]
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray) and base_value.ndim >= 1:
                base_value = float(base_value[1])
        else:
            shap_vals = np.array(shap_values)
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])

        return {
            'shap_values': shap_vals,
            'base_value': float(base_value),
            'X': X_sample,
            'feature_names': self.feature_names
        }

    def get_global_importance(
        self,
        X: Optional[np.ndarray] = None,
        method: str = 'mean_abs'
    ) -> Dict[str, float]:
        """
        Oblicz globalną ważność cech.

        Args:
            X: Dane (jeśli None, użyj X_background)
            method: Metoda agregacji ('mean_abs', 'mean', 'max')

        Returns:
            Słownik {nazwa_cechy: ważność}
        """
        X_data = X if X is not None else self.X_background

        result = self.explain_dataset(X_data)
        shap_vals = result['shap_values']

        if method == 'mean_abs':
            importance = np.abs(shap_vals).mean(axis=0)
        elif method == 'mean':
            importance = shap_vals.mean(axis=0)
        elif method == 'max':
            importance = np.abs(shap_vals).max(axis=0)
        else:
            raise ValueError(f"Nieznana metoda: {method}")

        importance_dict = dict(zip(self.feature_names, importance.tolist()))

        return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    def get_feature_ranking(
        self,
        explanation: Dict[str, Any]
    ) -> List[str]:
        """
        Uzyskaj ranking cech z wyjaśnienia.

        Args:
            explanation: Wyjaśnienie z explain_instance

        Returns:
            Lista nazw cech posortowanych wg ważności
        """
        return [fi['feature'] for fi in explanation['feature_impacts']]

    def plot_waterfall(
        self,
        explanation: Dict[str, Any],
        max_display: int = 10,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres waterfall dla pojedynczej instancji.

        Args:
            explanation: Wyjaśnienie z explain_instance
            max_display: Maksymalna liczba cech
            show: Czy wyświetlić
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        shap_values = np.array(explanation['shap_values'])
        feature_values = np.array(explanation['feature_values'])
        base_value = explanation['base_value']

        # Utwórz obiekt Explanation
        exp = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=feature_values,
            feature_names=self.feature_names
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        shap.plots.waterfall(exp, max_display=max_display, show=False)

        plt.title(
            f'SHAP Waterfall - Predykcja: {explanation["probability_positive"]:.1%} ryzyko',
            fontsize=12
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres waterfall zapisany do {save_path}")

        if show:
            plt.show()

        return plt.gcf()

    def plot_beeswarm(
        self,
        X: np.ndarray,
        max_display: int = 20,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres beeswarm (podsumowanie globalne).

        Args:
            X: Dane
            max_display: Maksymalna liczba cech
            show: Czy wyświetlić
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        result = self.explain_dataset(X)
        shap_vals = result['shap_values']

        fig, ax = plt.subplots(figsize=(10, 8))

        shap.summary_plot(
            shap_vals,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        plt.title('SHAP Summary - Wpływ cech na predykcję', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres beeswarm zapisany do {save_path}")

        if show:
            plt.show()

        return plt.gcf()

    def plot_bar(
        self,
        X: np.ndarray,
        max_display: int = 15,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres słupkowy globalnej ważności.

        Args:
            X: Dane
            max_display: Maksymalna liczba cech
            show: Czy wyświetlić
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        importance = self.get_global_importance(X)

        # Weź top N cech
        top_features = list(importance.items())[:max_display]
        features = [f[0] for f in top_features]
        values = [f[1] for f in top_features]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.invert_yaxis()

        ax.set_xlabel('Średnia |SHAP value|')
        ax.set_title('SHAP - Globalna ważność cech')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres bar zapisany do {save_path}")

        if show:
            plt.show()

        return fig

    def plot_force(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Wykres force plot dla pojedynczej instancji.

        Args:
            explanation: Wyjaśnienie z explain_instance
            save_path: Ścieżka do zapisu HTML
        """
        shap_values = np.array(explanation['shap_values'])
        base_value = explanation['base_value']
        feature_values = np.array(explanation['feature_values'])

        force_plot = shap.force_plot(
            base_value,
            shap_values,
            feature_values,
            feature_names=self.feature_names
        )

        if save_path:
            shap.save_html(save_path, force_plot)
            logger.info(f"Force plot zapisany do {save_path}")

        return force_plot

    def to_json(
        self,
        explanation: Dict[str, Any],
        top_n: int = 10
    ) -> str:
        """
        Serializuj wyjaśnienie do JSON dla LLM.

        Args:
            explanation: Wyjaśnienie
            top_n: Liczba top cech do uwzględnienia

        Returns:
            String JSON
        """
        json_exp = {
            'base_risk': explanation['base_value'],
            'predicted_risk': explanation['probability_positive'],
            'model_output_change': sum(explanation['shap_values']),
            'top_factors': explanation['feature_impacts'][:top_n],
            'summary': {
                'risk_factors_count': len(explanation['risk_factors']),
                'protective_factors_count': len(explanation['protective_factors']),
                'dominant_direction': 'risk' if len(explanation['risk_factors']) > len(explanation['protective_factors']) else 'protective'
            }
        }

        return json.dumps(json_exp, ensure_ascii=False, indent=2)

    def to_patient_friendly(
        self,
        explanation: Dict[str, Any],
        feature_translations: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Przekształć wyjaśnienie na format przyjazny dla pacjenta.

        Args:
            explanation: Wyjaśnienie
            feature_translations: Słownik tłumaczeń nazw cech

        Returns:
            Słownik z wyjaśnieniem dla pacjenta
        """
        translations = feature_translations or {
            'Wiek': 'Twój wiek',
            'Kreatynina': 'Poziom wskaźnika czynności nerek',
            'Manifestacja_Oddechowy': 'Stan układu oddechowego',
            'Manifestacja_Nos/Ucho/Gardlo': 'Objawy ze strony nosa, ucha lub gardła',
            'Max_CRP': 'Poziom stanu zapalnego',
            'Liczba_Zajetych_Narzadow': 'Liczba narządów objętych chorobą',
            'Manifestacja_Sercowo-Naczyniowy': 'Stan układu sercowo-naczyniowego',
            'Manifestacja_Nerki': 'Stan nerek',
            'Zaostrz_Wymagajace_OIT': 'Przebyte poważne zaostrzenia',
            'Dializa': 'Historia leczenia nerkozastępczego',
            'Plazmaferezy': 'Przebyte zabiegi oczyszczania krwi'
        }

        def translate(feature: str) -> str:
            return translations.get(feature, feature)

        prob = explanation['probability_positive']
        if prob < 0.3:
            risk_level = 'niski'
        elif prob < 0.7:
            risk_level = 'umiarkowany'
        else:
            risk_level = 'podwyższony'

        return {
            'risk_level': risk_level,
            'main_concerns': [
                translate(fi['feature'])
                for fi in explanation['risk_factors'][:3]
            ],
            'positive_factors': [
                translate(fi['feature'])
                for fi in explanation['protective_factors'][:3]
            ],
            'note': 'Omów te wyniki ze swoim lekarzem.'
        }
