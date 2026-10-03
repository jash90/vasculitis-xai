"""
Moduł DALEX Wrapper dla kompleksowej analizy modeli.

Implementuje wrapper dla biblioteki DALEX (Descriptive mAchine Learning EXplanations)
do wyjaśniania modeli klasyfikacji.
"""

import dalex as dx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DALEXWrapper:
    """
    Wrapper dla biblioteki DALEX.

    Zapewnia kompleksową analizę modeli ML z użyciem
    break down plots, SHAP, permutation importance i PDP.

    Attributes:
        explainer: Obiekt dalex.Explainer
        model: Model do wyjaśniania
        feature_names: Lista nazw cech
    """

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        label: str = 'Model',
        predict_function: Optional[callable] = None
    ):
        """
        Inicjalizacja DALEX Wrapper.

        Args:
            model: Wytrenowany model
            X: Dane (macierz cech)
            y: Etykiety
            feature_names: Lista nazw cech
            label: Etykieta modelu
            predict_function: Opcjonalna funkcja predykcji
        """
        self.model = model
        self.feature_names = feature_names
        self.label = label

        # Konwertuj X do DataFrame
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X, columns=feature_names)
        else:
            self.X = X

        self.y = y

        # Funkcja predykcji
        if predict_function is None:
            if hasattr(model, 'predict_proba'):
                predict_function = lambda m, d: m.predict_proba(d)[:, 1]
            else:
                predict_function = lambda m, d: m.predict(d)

        # Utwórz explainer DALEX
        self.explainer = dx.Explainer(
            model=model,
            data=self.X,
            y=y,
            label=label,
            predict_function=predict_function
        )

        logger.info(f"DALEXWrapper zainicjalizowany dla modelu '{label}'")

    def get_model_performance(
        self,
        model_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Oblicz metryki wydajności modelu.

        Args:
            model_type: Typ modelu ('classification' lub 'regression')

        Returns:
            Słownik z metrykami
        """
        mp = self.explainer.model_performance(model_type=model_type)

        result = mp.result.to_dict()

        logger.info(f"Model performance: AUC={result.get('auc', 'N/A')}")

        return result

    def explain_instance_break_down(
        self,
        instance: np.ndarray,
        order: Optional[List[str]] = None,
        interaction: bool = False
    ) -> Dict[str, Any]:
        """
        Wyjaśnienie Break Down dla pojedynczej instancji.

        Args:
            instance: Wektor cech
            order: Opcjonalna kolejność cech
            interaction: Czy uwzględnić interakcje

        Returns:
            Słownik z wyjaśnieniem
        """
        # Konwertuj do DataFrame
        if isinstance(instance, np.ndarray):
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            instance_df = pd.DataFrame(instance, columns=self.feature_names)
        else:
            instance_df = instance

        # Break Down
        if interaction:
            bd = self.explainer.predict_parts(
                instance_df,
                type='break_down_interactions',
                order=order
            )
        else:
            bd = self.explainer.predict_parts(
                instance_df,
                type='break_down',
                order=order
            )

        # Przetwórz wyniki
        result_df = bd.result

        contributions = []
        for _, row in result_df.iterrows():
            if row['variable'] not in ['intercept', 'prediction']:
                contributions.append({
                    'variable': row['variable_name'],
                    'variable_value': row['variable_value'],
                    'contribution': float(row['contribution']),
                    'cumulative': float(row['cumulative'])
                })

        # Predykcja
        # dalex: variable_name is '' for the prediction row, so match on `variable`
        intercept_row = result_df[result_df['variable'] == 'intercept']
        prediction_row = result_df[result_df['variable'] == 'prediction']

        explanation = {
            'intercept': float(intercept_row['contribution'].values[0]) if len(intercept_row) > 0 else 0,
            'prediction': float(prediction_row['cumulative'].values[0]) if len(prediction_row) > 0 else 0,
            'contributions': contributions,
            'type': 'break_down_interactions' if interaction else 'break_down'
        }

        # Podział na czynniki
        explanation['risk_factors'] = [
            c for c in contributions if c['contribution'] > 0
        ]
        explanation['protective_factors'] = [
            c for c in contributions if c['contribution'] < 0
        ]

        return explanation

    def explain_instance_shap(
        self,
        instance: np.ndarray,
        B: int = 25
    ) -> Dict[str, Any]:
        """
        Wyjaśnienie SHAP (przez DALEX) dla pojedynczej instancji.

        Args:
            instance: Wektor cech
            B: Liczba permutacji

        Returns:
            Słownik z wyjaśnieniem
        """
        # Konwertuj do DataFrame
        if isinstance(instance, np.ndarray):
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            instance_df = pd.DataFrame(instance, columns=self.feature_names)
        else:
            instance_df = instance

        # SHAP
        shap_exp = self.explainer.predict_parts(
            instance_df,
            type='shap',
            B=B
        )

        result_df = shap_exp.result

        # Agreguj wyniki (DALEX zwraca wiele wartości)
        contributions = {}
        for _, row in result_df.iterrows():
            var_name = row['variable_name']
            if var_name not in ['intercept', 'prediction', '']:
                if var_name not in contributions:
                    contributions[var_name] = []
                contributions[var_name].append(row['contribution'])

        # Uśrednij
        shap_values = {
            var: float(np.mean(vals))
            for var, vals in contributions.items()
        }

        # Posortuj wg bezwzględnej wartości
        sorted_shap = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanation = {
            'shap_values': shap_values,
            'sorted_features': [
                {'feature': feat, 'shap_value': val}
                for feat, val in sorted_shap
            ],
            'intercept': float(result_df[result_df['variable_name'] == 'intercept']['contribution'].mean()),
            'B': B
        }

        return explanation

    def get_variable_importance(
        self,
        loss_function: str = '1-auc',
        B: int = 10,
        variables: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Oblicz permutation importance dla cech.

        Args:
            loss_function: Funkcja straty do optymalizacji
            B: Liczba permutacji
            variables: Opcjonalna lista zmiennych

        Returns:
            Słownik {nazwa_cechy: ważność}
        """
        vi = self.explainer.model_parts(
            loss_function=loss_function,
            B=B,
            variables=variables,
            type='variable_importance'
        )

        result_df = vi.result
        full_model_loss = float(result_df.loc[result_df['variable'] == '_full_model_', 'dropout_loss'].iloc[0])

        # Importance = loss increase after permuting the feature (clipped at 0)
        importance = {}
        for _, row in result_df.iterrows():
            var_name = row['variable']
            if var_name not in ['_baseline_', '_full_model_']:
                importance[var_name] = max(0.0, float(row['dropout_loss']) - full_model_loss)

        # Posortuj
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"Top 5 ważnych cech: {list(importance_sorted.keys())[:5]}")

        return importance_sorted

    def get_partial_dependence(
        self,
        feature: str,
        grid_points: int = 101,
        variable_type: str = 'numerical'
    ) -> Dict[str, Any]:
        """
        Oblicz Partial Dependence Plot dla cechy.

        Args:
            feature: Nazwa cechy
            grid_points: Liczba punktów siatki
            variable_type: Typ zmiennej ('numerical' lub 'categorical')

        Returns:
            Słownik z danymi PDP
        """
        pdp = self.explainer.model_profile(
            variables=feature,
            N=None,  # Użyj wszystkich danych
            grid_points=grid_points,
            variable_type=variable_type
        )

        result_df = pdp.result

        pdp_data = {
            'feature': feature,
            'x_values': result_df['_x_'].tolist(),
            'y_values': result_df['_yhat_'].tolist(),
            'type': variable_type
        }

        return pdp_data

    def get_accumulated_local_effects(
        self,
        feature: str,
        grid_points: int = 101
    ) -> Dict[str, Any]:
        """
        Oblicz Accumulated Local Effects dla cechy.

        Args:
            feature: Nazwa cechy
            grid_points: Liczba punktów siatki

        Returns:
            Słownik z danymi ALE
        """
        ale = self.explainer.model_profile(
            variables=feature,
            N=None,
            grid_points=grid_points,
            type='accumulated'
        )

        result_df = ale.result

        ale_data = {
            'feature': feature,
            'x_values': result_df['_x_'].tolist(),
            'y_values': result_df['_yhat_'].tolist()
        }

        return ale_data

    def plot_break_down(
        self,
        explanation: Dict[str, Any],
        max_vars: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres Break Down.

        Args:
            explanation: Wyjaśnienie z explain_instance_break_down
            max_vars: Maksymalna liczba zmiennych
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        contributions = explanation['contributions'][:max_vars]

        fig, ax = plt.subplots(figsize=(10, 6))

        vars_names = [c['variable'] for c in contributions]
        values = [c['contribution'] for c in contributions]
        colors = ['#d73027' if v > 0 else '#1a9850' for v in values]

        y_pos = np.arange(len(vars_names))
        ax.barh(y_pos, values, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(vars_names, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Wkład do predykcji')
        ax.set_title(f'DALEX Break Down - Predykcja: {explanation["prediction"]:.3f}')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres Break Down zapisany do {save_path}")

        return fig

    def plot_variable_importance(
        self,
        importance: Dict[str, float],
        max_vars: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres ważności zmiennych.

        Args:
            importance: Słownik z ważnością cech
            max_vars: Maksymalna liczba zmiennych
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        # Weź top N
        top_importance = dict(list(importance.items())[:max_vars])

        fig, ax = plt.subplots(figsize=(10, 6))

        vars_names = list(top_importance.keys())
        values = list(top_importance.values())

        y_pos = np.arange(len(vars_names))
        ax.barh(y_pos, values, color='steelblue', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(vars_names, fontsize=9)
        ax.invert_yaxis()

        ax.set_xlabel('Dropout Loss')
        ax.set_title('DALEX - Permutation Variable Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres VI zapisany do {save_path}")

        return fig

    def plot_partial_dependence(
        self,
        feature: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Wykres Partial Dependence.

        Args:
            feature: Nazwa cechy
            save_path: Ścieżka do zapisu

        Returns:
            Figura matplotlib
        """
        pdp_data = self.get_partial_dependence(feature)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(pdp_data['x_values'], pdp_data['y_values'], linewidth=2, color='steelblue')
        ax.fill_between(pdp_data['x_values'], pdp_data['y_values'], alpha=0.3)

        ax.set_xlabel(feature)
        ax.set_ylabel('Średnia predykcja')
        ax.set_title(f'Partial Dependence Plot - {feature}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wykres PDP zapisany do {save_path}")

        return fig

    def to_json(self, explanation: Dict[str, Any]) -> str:
        """
        Serializuj wyjaśnienie do JSON.

        Args:
            explanation: Wyjaśnienie

        Returns:
            String JSON
        """
        return json.dumps(explanation, ensure_ascii=False, indent=2)

    def create_arena(
        self,
        port: int = 8002
    ):
        """
        Uruchom interaktywny dashboard Arena.

        Args:
            port: Port do nasłuchiwania

        Returns:
            Obiekt Arena
        """
        arena = dx.Arena()
        arena.push_model(self.explainer)
        arena.run_server(port=port)

        return arena

    def compare_models(
        self,
        other_explainers: List['DALEXWrapper']
    ) -> pd.DataFrame:
        """
        Porównaj wiele modeli.

        Args:
            other_explainers: Lista innych wrapperów DALEX

        Returns:
            DataFrame z porównaniem
        """
        all_explainers = [self.explainer] + [w.explainer for w in other_explainers]

        # Porównaj wydajność
        performances = []
        for exp in all_explainers:
            mp = exp.model_performance()
            perf = mp.result.copy()
            perf['model'] = exp.label
            performances.append(perf)

        comparison_df = pd.concat(performances, ignore_index=True)

        return comparison_df
