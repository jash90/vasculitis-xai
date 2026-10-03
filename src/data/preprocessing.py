"""
Moduł preprocessingu danych dla systemu XAI.

Zawiera klasę DataPreprocessor do wczytywania, czyszczenia i przygotowania
danych pacjentów z zapaleniem naczyń do analizy ML.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Kluczowe cechy kliniczne zidentyfikowane w analizie
KLUCZOWE_CECHY = [
    'Wiek', 'Plec', 'Wiek_rozpoznania',
    'Opoznienie_Rozpoznia',
    'Liczba_Zajetych_Narzadow',
    'Manifestacja_Sercowo-Naczyniowy',
    'Manifestacja_Nerki',
    'Manifestacja_Pokarmowy',
    'Manifestacja_Zajecie_CSN',
    'Manifestacja_Neurologiczny',
    'Zaostrz_Wymagajace_OIT',
    'Kreatynina', 'Max_CRP',
    'Plazmaferezy', 'Dializa',
    'Sterydy_Dawka_g', 'Czas_Sterydow',
    'Powiklania_Serce/pluca',
    'Powiklania_Infekcja'
]


class DataPreprocessor:
    """
    Klasa do preprocessingu danych pacjentów z zapaleniem naczyń.

    Attributes:
        scaler: Obiekt skalera (StandardScaler lub MinMaxScaler)
        encoders: Słownik enkoderów dla zmiennych kategorycznych
        selected_features: Lista wybranych cech
        imputer: Obiekt do uzupełniania brakujących wartości
        feature_names: Lista nazw cech po preprocessingu
    """

    def __init__(self):
        """Inicjalizacja preprocessora."""
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.selected_features: Optional[List[str]] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_names: Optional[List[str]] = None
        self._is_fitted: bool = False

    def load_data(self, filepath: str, separator: str = '|') -> pd.DataFrame:
        """
        Wczytaj dane z pliku CSV.

        Args:
            filepath: Ścieżka do pliku CSV
            separator: Separator kolumn (domyślnie '|' dla aktualne_dane.csv)

        Returns:
            DataFrame z wczytanymi danymi
        """
        logger.info(f"Wczytywanie danych z {filepath}")

        df = pd.read_csv(filepath, sep=separator)

        # Usuń kolumnę identyfikatora jeśli istnieje
        id_columns = ['Kod', 'ID', 'id', 'Patient_ID']
        for col in id_columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
                logger.info(f"Usunięto kolumnę identyfikatora: {col}")

        logger.info(f"Wczytano {len(df)} rekordów z {len(df.columns)} kolumnami")
        return df

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generuj podsumowanie danych.

        Args:
            df: DataFrame do analizy

        Returns:
            Słownik z podsumowaniem
        """
        summary = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(exclude=[np.number]).columns.tolist(),
        }

        # Sprawdź wartości -1 (często oznaczają brakujące dane)
        minus_one_counts = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            count = (df[col] == -1).sum()
            if count > 0:
                minus_one_counts[col] = count
        summary['minus_one_values'] = minus_one_counts

        return summary

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'median',
        treat_minus_one_as_missing: bool = True
    ) -> pd.DataFrame:
        """
        Obsłuż brakujące wartości.

        Args:
            df: DataFrame z danymi
            strategy: Strategia uzupełniania ('median', 'mean', 'mode', 'constant')
            treat_minus_one_as_missing: Czy traktować -1 jako brakujące

        Returns:
            DataFrame z uzupełnionymi wartościami
        """
        df = df.copy()

        # Zamień -1 na NaN jeśli włączone
        if treat_minus_one_as_missing:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                minus_one_count = (df[col] == -1).sum()
                if minus_one_count > 0:
                    df[col] = df[col].replace(-1, np.nan)
                    logger.info(f"Zamieniono {minus_one_count} wartości -1 na NaN w kolumnie {col}")

        # Uzupełnij brakujące wartości
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    if strategy == 'median':
                        fill_value = df[col].median()
                    elif strategy == 'mean':
                        fill_value = df[col].mean()
                    elif strategy == 'mode':
                        fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                    else:
                        fill_value = 0
                    df[col] = df[col].fillna(fill_value)
                else:
                    # Dla kolumn kategorycznych użyj mody
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col] = df[col].fillna(mode_val)

                logger.info(f"Uzupełniono {missing_count} brakujących wartości w kolumnie {col}")

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Zakoduj zmienne kategoryczne.

        Args:
            df: DataFrame z danymi
            columns: Lista kolumn do zakodowania (None = automatyczne wykrycie)

        Returns:
            DataFrame z zakodowanymi zmiennymi
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
                logger.info(f"Zakodowano kolumnę {col}")

        return df

    def remove_high_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Usuń cechy o wysokiej korelacji.

        Args:
            df: DataFrame z danymi
            threshold: Próg korelacji do usunięcia
            target_col: Kolumna docelowa (nie usuwać)

        Returns:
            DataFrame z usuniętymi skorelowanymi cechami
        """
        df = df.copy()

        # Wybierz tylko kolumny numeryczne (bez target)
        if target_col and target_col in df.columns:
            numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        # Oblicz macierz korelacji
        corr_matrix = numeric_df.corr().abs()

        # Znajdź pary o wysokiej korelacji
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Kolumny do usunięcia
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        if to_drop:
            logger.info(f"Usuwanie {len(to_drop)} wysoko skorelowanych cech: {to_drop}")
            df = df.drop(columns=to_drop)

        return df

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20,
        method: str = 'mutual_info'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Wybierz najważniejsze cechy.

        Args:
            X: DataFrame z cechami
            y: Zmienna docelowa
            n_features: Liczba cech do wyboru
            method: Metoda selekcji ('mutual_info', 'f_classif')

        Returns:
            Tuple (DataFrame z wybranymi cechami, lista nazw cech)
        """
        # Upewnij się że mamy tylko kolumny numeryczne
        X_numeric = X.select_dtypes(include=[np.number])

        # Ogranicz n_features do dostępnej liczby cech
        n_features = min(n_features, X_numeric.shape[1])

        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            selector = SelectKBest(score_func=f_classif, k=n_features)

        X_selected = selector.fit_transform(X_numeric, y)

        # Pobierz nazwy wybranych cech
        selected_mask = selector.get_support()
        self.selected_features = X_numeric.columns[selected_mask].tolist()

        # Pobierz wyniki selekcji
        scores = selector.scores_
        feature_scores = dict(zip(X_numeric.columns, scores))
        feature_scores_sorted = dict(
            sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"Wybrano {n_features} cech metodą {method}")
        logger.info(f"Top 10 cech: {list(feature_scores_sorted.keys())[:10]}")

        return pd.DataFrame(X_selected, columns=self.selected_features), self.selected_features

    def scale_features(
        self,
        X: np.ndarray,
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """
        Skaluj cechy.

        Args:
            X: Macierz cech
            method: Metoda skalowania ('standard', 'minmax')
            fit: Czy dopasować scaler (True dla danych treningowych)

        Returns:
            Przeskalowana macierz cech
        """
        if method == 'standard':
            if self.scaler is None or fit:
                self.scaler = StandardScaler()
        else:
            if self.scaler is None or fit:
                self.scaler = MinMaxScaler()

        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def get_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.0,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple:
        """
        Podziel dane na zbiory treningowy, walidacyjny i testowy.

        Args:
            X: Macierz cech
            y: Wektor etykiet
            test_size: Rozmiar zbioru testowego
            val_size: Rozmiar zbioru walidacyjnego (0 = brak)
            stratify: Czy zachować proporcje klas
            random_state: Ziarno losowości

        Returns:
            Tuple z podziałem danych
        """
        stratify_param = y if stratify else None

        # Podział na train i test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_param,
            random_state=random_state
        )

        if val_size > 0:
            # Oblicz proporcję walidacji względem pozostałych danych
            val_ratio = val_size / (1 - test_size)
            stratify_param = y_train if stratify else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_ratio,
                stratify=stratify_param,
                random_state=random_state
            )

            logger.info(f"Podział danych: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test

        logger.info(f"Podział danych: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def prepare_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = 'Zgon',
        n_features: int = 20,
        scale: bool = True,
        handle_missing: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Pełny pipeline preprocessingu.

        UWAGA (przestarzałe, przeciek danych): imputacja, filtr korelacji,
        selekcja cech (z użyciem y) i skalowanie są dopasowywane do CAŁEGO
        przekazanego zbioru, zanim nastąpi podział train/test. Kod 0 nie jest
        traktowany jako brak danych. Do nowych modeli używaj src.data.cohort
        + pipeline'ów z src.models.model_zoo (dopasowanie wewnątrz foldów).

        Args:
            df: DataFrame z danymi
            target_col: Nazwa kolumny docelowej
            n_features: Liczba cech do wyboru
            scale: Czy skalować cechy
            handle_missing: Czy obsłużyć brakujące wartości

        Returns:
            Tuple (X, y, feature_names)
        """
        df = df.copy()

        # Obsłuż brakujące wartości
        if handle_missing:
            df = self.handle_missing_values(df)

        # Zakoduj zmienne kategoryczne
        df = self.encode_categorical(df)

        # Usuń wysoko skorelowane cechy
        df = self.remove_high_correlation(df, target_col=target_col)

        # Wyodrębnij X i y
        if target_col not in df.columns:
            raise ValueError(f"Kolumna docelowa '{target_col}' nie istnieje w danych")

        y = df[target_col].values
        X = df.drop(target_col, axis=1)

        # Selekcja cech
        X, self.feature_names = self.select_features(X, y, n_features=n_features)
        X = X.values

        # Skalowanie
        if scale:
            X = self.scale_features(X)

        self._is_fitted = True

        logger.info(f"Pipeline zakończony: {X.shape[0]} próbek, {X.shape[1]} cech")
        return X, y, self.feature_names

    def transform(self, df: pd.DataFrame, target_col: str = 'Zgon') -> np.ndarray:
        """
        Przekształć nowe dane używając dopasowanego pipeline'a.

        Args:
            df: DataFrame z nowymi danymi
            target_col: Nazwa kolumny docelowej (do usunięcia jeśli istnieje)

        Returns:
            Przekształcona macierz cech
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline nie został dopasowany. Użyj najpierw prepare_pipeline().")

        df = df.copy()

        # Obsłuż brakujące wartości
        df = self.handle_missing_values(df)

        # Zakoduj zmienne kategoryczne (używając istniejących enkoderów)
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        # Usuń kolumnę docelową jeśli istnieje
        if target_col in df.columns:
            df = df.drop(target_col, axis=1)

        # Wybierz tylko wybrane cechy
        if self.selected_features:
            missing_features = set(self.selected_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Brakujące cechy w danych: {missing_features}")
            df = df[self.selected_features]

        # Skaluj
        if self.scaler:
            return self.scaler.transform(df.values)

        return df.values

    def save_config(self, filepath: str) -> None:
        """
        Zapisz konfigurację preprocessora.

        Args:
            filepath: Ścieżka do pliku JSON
        """
        config = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'is_fitted': self._is_fitted
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"Zapisano konfigurację do {filepath}")

    def load_config(self, filepath: str) -> None:
        """
        Wczytaj konfigurację preprocessora.

        Args:
            filepath: Ścieżka do pliku JSON
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.feature_names = config.get('feature_names')
        self.selected_features = config.get('selected_features')
        self._is_fitted = config.get('is_fitted', False)

        logger.info(f"Wczytano konfigurację z {filepath}")
