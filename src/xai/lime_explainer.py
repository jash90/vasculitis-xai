"""
Moduł LIME Explainer dla wyjaśnień lokalnych.

Implementuje Local Interpretable Model-agnostic Explanations (LIME)
dla modeli klasyfikacji ryzyka śmiertelności.
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any, Optional, Union, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Klasa do generowania wyjaśnień LIME.

    Zapewnia lokalne wyjaśnienia predykcji modeli ML
    używając metody LIME (Local Interpretable Model-agnostic Explanations).

    Attributes:
        model: Model do wyjaśniania
        feature_names: Lista nazw cech
        class_names: Nazwy klas
        explainer: Obiekt LimeTabularExplainer
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: List[str],
        class_names: List[str] = None,
        mode: str = 'classification',
        categorical_features: Optional[List[int]] = None,
        discretize_continuous: bool = True,
        random_state: int = 42
    ):
        """
        Inicjalizacja LIME Explainera.

        Args:
            model: Wytrenowany model
            X_train: Dane treningowe (do obliczenia statystyk)
            feature_names: Lista nazw cech
            class_names: Nazwy klas (domyślnie ['Przeżycie', 'Zgon'])
            mode: Tryb ('classification' lub 'regression')
            categorical_features: Indeksy cech kategorycznych
            discretize_continuous: Czy dyskretyzować ciągłe
            random_state: Ziarno losowości
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Przeżycie', 'Zgon']
        self.mode = mode
        self.random_state = random_state

        # Funkcja predykcji
        if hasattr(model, 'predict_proba'):
            self.predict_fn = model.predict_proba
        else:
            # Fallback dla modeli bez predict_proba
            def predict_fn(X):
                preds = model.predict(X)
                return np.column_stack([1 - preds, preds])
            self.predict_fn = predict_fn

        # Inicjalizacja explainera
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            mode=mode,
            categorical_features=categorical_features,
            discretize_continuous=discretize_continuous,
            random_state=random_state
        )

        logger.info(f"LIMEExplainer zainicjalizowany dla {len(feature_names)} cech")

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        labels: tuple = (1,)
    ) -> Dict[str, Any]:
        """
        Wygeneruj wyjaśnienie dla pojedynczej instancji.

        Args:
            instance: Wektor cech (1D array)
            num_features: Liczba cech do wyświetlenia
            num_samples: Liczba próbek perturbacji
            labels: Klasy do wyjaśnienia (domyślnie klasa pozytywna)

        Returns:
            Słownik z wyjaśnieniem
        """
        # Upewnij się że instance jest 1D
        if instance.ndim > 1:
            instance = instance.flatten()

        # Wygeneruj wyjaśnienie
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=labels
        )

        # Predykcja dla tej instancji
        proba = self.predict_fn(instance.reshape(1, -1))[0]
        prediction = int(np.argmax(proba))

        # Wyodrębnij wyjaśnienie
        label = labels[0]  # Klasa do wyjaśnienia
        feature_weights = exp.as_list(label=label)

        # Posortuj wg bezwzględnej wartości wagi
        feature_weights_sorted = sorted(
            feature_weights,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Przygotuj wynik
        explanation = {
            'prediction': prediction,
            'prediction_label': self.class_names[prediction],
            'probability': {
                self.class_names[0]: float(proba[0]),
                self.class_names[1]: float(proba[1])
            },
            'probability_positive': float(proba[1]),
            'feature_weights': feature_weights_sorted,
            'intercept': float(exp.intercept[label]),
            'local_prediction': float(exp.local_pred[0]),
            'num_features': num_features,
            'num_samples': num_samples,
            'instance_values': dict(zip(self.feature_names, instance.tolist()))
        }

        # Podział na czynniki ryzyka i ochronne
        risk_factors = [
            (feat, weight) for feat, weight in feature_weights_sorted
            if weight > 0
        ]
        protective_factors = [
            (feat, weight) for feat, weight in feature_weights_sorted
            if weight < 0
        ]

        explanation['risk_factors'] = risk_factors
        explanation['protective_factors'] = protective_factors

        return explanation

    def explain_batch(
        self,
        instances: np.ndarray,
        num_features: int = 10,
        num_samples: int = 3000
    ) -> List[Dict[str, Any]]:
        """
        Wygeneruj wyjaśnienia dla wielu instancji.

        Args:
            instances: Macierz cech (2D array)
            num_features: Liczba cech do wyświetlenia
            num_samples: Liczba próbek perturbacji

        Returns:
            Lista wyjaśnień
        """
        explanations = []

        for i, instance in enumerate(instances):
            logger.info(f"Generowanie wyjaśnienia {i+1}/{len(instances)}")
            exp = self.explain_instance(
                instance,
                num_features=num_features,
                num_samples=num_samples
            )
            explanations.append(exp)

        return explanations

    def get_feature_importance(
        self,
        explanation: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Wyodrębnij ważność cech z wyjaśnienia.

        Args:
            explanation: Wyjaśnienie z explain_instance

        Returns:
            Słownik {nazwa_cechy: waga}
        """
        importance = {}

        for feature_desc, weight in explanation['feature_weights']:
            # Wyodrębnij nazwę cechy z opisu (np. "Wiek > 50" -> "Wiek")
            for feature_name in self.feature_names:
                if feature_name in feature_desc:
                    importance[feature_name] = weight
                    break
            else:
                importance[feature_desc] = weight

        return importance

    def get_feature_ranking(
        self,
        explanation: Dict[str, Any],
        absolute: bool = True
    ) -> List[str]:
        """
        Uzyskaj ranking cech z wyjaśnienia.

        Args:
            explanation: Wyjaśnienie z explain_instance
            absolute: Czy sortować wg wartości bezwzględnych

        Returns:
            Lista nazw cech posortowanych wg ważności
        """
        importance = self.get_feature_importance(explanation)

        if absolute:
            sorted_features = sorted(
                importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        else:
            sorted_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

        return [feat for feat, _ in sorted_features]

    def calculate_stability(
        self,
        instance: np.ndarray,
        n_runs: int = 100,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Zbadaj stabilność wyjaśnień LIME.

        Args:
            instance: Instancja do wyjaśnienia
            n_runs: Liczba powtórzeń
            num_features: Liczba cech

        Returns:
            Metryki stabilności
        """
        all_weights = {feat: [] for feat in self.feature_names}
        all_rankings = []

        for i in range(n_runs):
            exp = self.explain_instance(
                instance,
                num_features=num_features,
                num_samples=1000  # Mniejsza liczba dla szybkości
            )

            # Zapisz wagi
            importance = self.get_feature_importance(exp)
            for feat in self.feature_names:
                all_weights[feat].append(importance.get(feat, 0))

            # Zapisz ranking
            ranking = self.get_feature_ranking(exp)[:5]  # Top 5
            all_rankings.append(ranking)

        # Oblicz metryki stabilności
        stability_metrics = {
            'weight_std': {
                feat: np.std(weights) for feat, weights in all_weights.items()
            },
            'weight_mean': {
                feat: np.mean(weights) for feat, weights in all_weights.items()
            },
            'top_feature_consistency': self._calculate_ranking_consistency(all_rankings),
            'n_runs': n_runs
        }

        # Średnia niestabilność (std/|mean|)
        instabilities = []
        for feat in self.feature_names:
            mean_val = abs(stability_metrics['weight_mean'][feat])
            if mean_val > 0.001:
                instability = stability_metrics['weight_std'][feat] / mean_val
                instabilities.append(instability)

        stability_metrics['mean_instability'] = np.mean(instabilities) if instabilities else 0

        return stability_metrics

    def _calculate_ranking_consistency(
        self,
        rankings: List[List[str]]
    ) -> float:
        """
        Oblicz zgodność rankingów (jak często ta sama cecha jest na pierwszym miejscu).
        """
        if not rankings:
            return 0.0

        top_features = [r[0] if r else None for r in rankings]
        most_common = max(set(top_features), key=top_features.count)
        consistency = top_features.count(most_common) / len(top_features)

        return consistency

    def plot_explanation(
        self,
        explanation: Dict[str, Any],
        max_features: int = 10,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wizualizacja wyjaśnienia LIME.

        Args:
            explanation: Wyjaśnienie z explain_instance
            max_features: Maksymalna liczba cech do wyświetlenia
            figsize: Rozmiar figury
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        feature_weights = explanation['feature_weights'][:max_features]

        features = [fw[0] for fw in feature_weights]
        weights = [fw[1] for fw in feature_weights]

        # Kolory
        colors = ['#d73027' if w > 0 else '#1a9850' for w in weights]

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(features))
        ax.barh(y_pos, weights, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.invert_yaxis()  # Najważniejsza cecha na górze

        ax.set_xlabel('Waga (wpływ na predykcję)')
        ax.set_title(
            f'LIME - Wyjaśnienie predykcji\n'
            f'Predykcja: {explanation["prediction_label"]} '
            f'(prawdopodobieństwo: {explanation["probability_positive"]:.1%})'
        )

        # Dodaj legendę
        ax.axvline(x=0, color='black', linewidth=0.5)

        # Dodaj adnotację
        risk_count = len(explanation['risk_factors'])
        protective_count = len(explanation['protective_factors'])
        ax.text(
            0.02, 0.98,
            f'Czynniki ryzyka: {risk_count}\nCzynniki ochronne: {protective_count}',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres LIME zapisany do {save_path}")

        return fig

    def to_json(self, explanation: Dict[str, Any]) -> str:
        """
        Serializuj wyjaśnienie do JSON.

        Args:
            explanation: Wyjaśnienie

        Returns:
            String JSON
        """
        # Konwertuj do formatu JSON-friendly
        json_exp = {
            'prediction': explanation['prediction'],
            'prediction_label': explanation['prediction_label'],
            'probability': explanation['probability'],
            'feature_contributions': [
                {
                    'feature': feat,
                    'contribution': float(weight),
                    'direction': 'zwiększa ryzyko' if weight > 0 else 'zmniejsza ryzyko'
                }
                for feat, weight in explanation['feature_weights']
            ],
            'top_risk_factors': [
                {'feature': feat, 'contribution': float(w)}
                for feat, w in explanation['risk_factors'][:5]
            ],
            'top_protective_factors': [
                {'feature': feat, 'contribution': float(w)}
                for feat, w in explanation['protective_factors'][:5]
            ],
            'intercept': explanation['intercept']
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
            'Max_CRP': 'Poziom stanu zapalnego w organizmie',
            'Liczba_Zajetych_Narzadow': 'Liczba narządów objętych chorobą',
            'Manifestacja_Sercowo-Naczyniowy': 'Stan układu krążenia',
            'Manifestacja_Nerki': 'Stan nerek',
            'Zaostrz_Wymagajace_OIT': 'Przebyte poważne zaostrzenia',
            'Dializa': 'Potrzeba dializy',
            'Plazmaferezy': 'Przebyte plazmaferezy'
        }

        def translate_feature(feat_desc: str) -> str:
            for orig, trans in translations.items():
                if orig in feat_desc:
                    return feat_desc.replace(orig, trans)
            return feat_desc

        # Określ poziom ryzyka
        prob = explanation['probability_positive']
        if prob < 0.3:
            risk_level = 'niski'
            risk_desc = 'Analiza wskazuje na niskie ryzyko.'
        elif prob < 0.7:
            risk_level = 'umiarkowany'
            risk_desc = 'Analiza wskazuje na umiarkowane ryzyko.'
        else:
            risk_level = 'podwyższony'
            risk_desc = 'Analiza wskazuje na podwyższone ryzyko.'

        patient_explanation = {
            'risk_level': risk_level,
            'risk_description': risk_desc,
            'main_concerns': [
                translate_feature(feat)
                for feat, _ in explanation['risk_factors'][:3]
            ],
            'positive_factors': [
                translate_feature(feat)
                for feat, _ in explanation['protective_factors'][:3]
            ],
            'recommendation': 'Omów te wyniki ze swoim lekarzem prowadzącym.'
        }

        return patient_explanation
