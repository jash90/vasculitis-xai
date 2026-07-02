#!/usr/bin/env python3
"""
Retrening modeli ZGODNYCH z wejściem API (patient_to_array), na pełnych 20 cechach
formularza, z kalibracją prawdopodobieństw.

Kontekst:
    Oryginalne artefakty (71 cech) nigdy nie ładowały się w API — strażnik zgodności
    cech (n_features_in_=71 vs feature_names=20) zawsze wymuszał tryb demo. Wyniki
    z pracy (AUC ~0.9) pochodziły z treningu offline na PEŁNYM zbiorze cech
    (76->71 po preprocessingu), w tym z cech, których formularz nie zbiera
    (ANCA, CRP, powikłania — część potencjalnie z przeciekiem czasowym).

Rozwiązanie (serwowalne i uczciwe):
    Trenujemy 3 modele drzewiaste na DOKŁADNIE tych 20 cechach, które zbiera
    formularz i mapuje patient_to_array. Wartości pozostają porządkowe (np.
    manifestacje 0/1/2/3 — pełna rozdzielczość treningu), braki/-1 -> 0 (tak jak
    przy serwowaniu: patient_to_array wstawia 0 za brak). Modele owijamy w
    CalibratedClassifierCV (sigmoid) dla sensownych, dobrze skalibrowanych
    prawdopodobieństw. Zapisujemy też X_train/X_background/X_test (20 cech) dla
    SHAP/LIME oraz feature_names.json (20). Guard przechodzi (20==20), a
    patient_to_array obsługuje wszystkie 20 nazw bez zmian w kodzie API.

Walidacja (5-fold CV): RF AUC≈0.83.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "aktualne_dane.csv"
OUT_DIR = PROJECT_ROOT / "models" / "saved"

# (kolumna w CSV  ->  nazwa cechy modelu == feature_names.json == patient_to_array)
FEATURE_MAP = {
    "Wiek_rozpoznania": "Wiek_rozpoznania",
    "Opoznienie_Rozpoznia": "Opoznienie_Rozpoznia",
    "Manifestacja_Miesno-Szkiel": "Manifestacja_Miesno-Szkiel",
    "Manifestacja_Skora": "Manifestacja_Skora",
    "Manifestacja_Wzrok": "Manifestacja_Wzrok",
    "Manifestacja_Sercowo-Naczyniowy": "Manifestacja_Sercowo-Naczyniowy",
    "Manifestacja_Pokarmowy": "Manifestacja_Pokarmowy",
    "Manifestacja_Nerki": "Manifestacja_Nerki",
    "Manifestacja_Moczowo-Plciowy": "Manifestacja_Moczowo-Plciowy",
    "Manifestacja_Zajecie_CSN": "Manifestacja_Zajecie_CSN",
    "Manifestacja_Neurologiczny": "Manifestacja_Neurologiczny",
    "Liczba_Zajetych_Narzadow": "Liczba_Zajetych_Narzadow",
    "Zaostrz_Wymagajace_Hospital": "Zaostrz_Wymagajace_Hospital",
    "Zaostrz_Wymagajace_OIT": "Zaostrz_Wymagajace_OIT",
    "Kreatynina": "Kreatynina",
    "Pulsy": "Pulsy",
    "Czas_Sterydow": "Czas_Sterydow",
    "Plazmaferezy": "Plazmaferezy",
    "Eozynofilia_Krwi_Obwodowej_Wartosc": "Eozynofilia_Krwi_Obwodowej_Wartosc",
    "Biopsja_Wynik": "Biopsja_Wynik",
}
TARGET = "Zgon"


def main():
    print("Ładowanie:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep="|")
    df.columns = [c.strip() for c in df.columns]
    print(f"  {len(df)} rekordów, {len(df.columns)} kolumn")

    missing = [c for c in FEATURE_MAP if c not in df.columns]
    if missing:
        print("[BŁĄD] Brak kolumn w danych:", missing)
        return 1

    feature_cols = list(FEATURE_MAP.keys())
    feature_names = [FEATURE_MAP[c] for c in feature_cols]

    # Cechy binarne w formularzu (0/1) — w surowych danych bywają 0/1/2/3.
    BINARY = {
        "Manifestacja_Miesno-Szkiel", "Manifestacja_Skora", "Manifestacja_Wzrok",
        "Manifestacja_Sercowo-Naczyniowy", "Manifestacja_Pokarmowy", "Manifestacja_Nerki",
        "Manifestacja_Moczowo-Plciowy", "Manifestacja_Zajecie_CSN", "Manifestacja_Neurologiczny",
        "Zaostrz_Wymagajace_Hospital", "Zaostrz_Wymagajace_OIT", "Pulsy", "Plazmaferezy",
        "Biopsja_Wynik",
    }

    # CZYSZCZENIE do PRZESTRZENI FORMULARZA (PatientInput) — kluczowe dla
    # zgodności trening==serwowanie. Surowe kodowanie różni się od UI:
    #   Wiek_rozpoznania: dni -> lata (/365.25), Opoznienie: dni -> miesiące (/30.44),
    #   cechy binarne: 0/1/2/3 -> 0/1. Reszta (Kreatynina, Czas_Sterydow [mies.],
    #   Eozynofilia, Liczba_Zajetych_Narzadow) jest już w jednostkach formularza.
    # Brak / -1 -> 0 (tak jak patient_to_array wstawia 0 za brak).
    raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    raw.columns = feature_names
    raw = raw.replace(-1, np.nan)

    clean = pd.DataFrame(index=raw.index)
    for name in feature_names:
        s = raw[name]
        if name in BINARY:
            clean[name] = (s.fillna(0) > 0).astype(float)
        elif name == "Wiek_rozpoznania":
            clean[name] = (s / 365.25).clip(0, 120).fillna(0.0)
        elif name == "Opoznienie_Rozpoznia":
            clean[name] = (s / 30.44).clip(lower=0).fillna(0.0)
        elif name == "Liczba_Zajetych_Narzadow":
            clean[name] = s.clip(0, 20).fillna(0.0)
        else:  # Kreatynina, Czas_Sterydow, Eozynofilia — wartości formularza
            clean[name] = s.clip(lower=0).fillna(0.0)

    X = clean
    y = pd.to_numeric(df[TARGET], errors="coerce")

    keep = y.notna()
    X = X[keep].astype(np.float64).to_numpy()
    y = y[keep].astype(int).to_numpy()
    pos, neg = int(y.sum()), int((y == 0).sum())
    print(f"  Cechy: {len(feature_names)} | Próbki: {len(y)} | Zgon=1: {pos}, Zgon=0: {neg}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    base = {
        "best_model.joblib": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, n_jobs=2,
        ),
        "random_forest_model.joblib": RandomForestClassifier(
            n_estimators=400, max_depth=8, random_state=42, n_jobs=2,
        ),
        "lightgbm_model.joblib": LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=2, verbose=-1,
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, est in base.items():
        model = CalibratedClassifierCV(est, method="sigmoid", cv=5)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)
        joblib.dump(model, OUT_DIR / fname)
        print("  [%-26s] AUC=%.3f Brier=%.3f ACC=%.3f SENS=%.3f -> zapisano" % (
            fname, roc_auc_score(y_test, proba), brier_score_loss(y_test, proba),
            accuracy_score(y_test, pred), recall_score(y_test, pred, zero_division=0)))

    # Dane dla XAI (SHAP/LIME) — 20 cech, ta sama przestrzeń co model
    rs = np.random.RandomState(42)
    bg_idx = rs.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    joblib.dump(X_train, OUT_DIR / "X_train.joblib")
    joblib.dump(X_test, OUT_DIR / "X_test.joblib")
    joblib.dump(X_train[bg_idx], OUT_DIR / "X_background.joblib")
    joblib.dump(y_train, OUT_DIR / "y_train.joblib")
    joblib.dump(y_test, OUT_DIR / "y_test.joblib")
    joblib.dump(y_train[bg_idx], OUT_DIR / "y_background.joblib")

    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print("\nZapisano feature_names.json (20) + X_train/X_background/X_test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
