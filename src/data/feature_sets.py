"""
Feature sets allowed at the prediction time "at diagnosis".

Anything recorded during follow-up (flares, 6-month treatment, complications,
disease course, steroid duration/cumulative dose, age at last contact) is
blocked, because it is not known when the prediction is made and partly
encodes the outcome itself.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .cohort import DIALYSIS_TARGET, MORTALITY_TARGET

POST_BASELINE_PREFIXES = (
    "Leczenie_6m_",
    "Powiklania_",
    "Powiklanie_",
    "Zaostrz_",
)
POST_BASELINE_COLUMNS = {
    "Wiek",  # age at last contact / death, not at diagnosis
    "Liczba_Zaostrzen",
    "Czas_Pierwsze_Zaostrzenie",
    "Przebieg_scalony",
    "Czas_Sterydow",
    "Sterydy_Dawka_g",
    "Sterydy_Dawka_mg",
}

# Features whose timing is plausible-at-diagnosis but not documented; reported
# for confirmation with the data owner.
UNCERTAIN_TIMING = (
    "Max_CRP",
    "Kreatynina",
    "Pulsy",
    "Plazmaferezy",
    "Droga_Podania",
    "Leczenie_*",
)

CATEGORICAL_FEATURES = ("Rozpoznanie", "Praca", "Palenie_Tytoniu", "Pulsy", "Droga_Podania", "Biopsja_Wynik")


def is_post_baseline(column: str) -> bool:
    return column in POST_BASELINE_COLUMNS or column.startswith(POST_BASELINE_PREFIXES)


def baseline_features(columns: Iterable[str], task: str) -> List[str]:
    """Columns usable for `task` ('mortality' or 'dialysis') at diagnosis."""
    if task not in ("mortality", "dialysis"):
        raise ValueError(f"Unknown task: {task}")
    # Dialysis is follow-up treatment (post-baseline for mortality) and
    # death is post-baseline for dialysis, so both targets are always excluded.
    excluded = {MORTALITY_TARGET, DIALYSIS_TARGET}
    return [c for c in columns if c not in excluded and not is_post_baseline(c)]


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features (no fitting, safe before CV).

    eGFR uses the MDRD form without the sex factor because the coding of `Plec`
    (0/1) is not documented; it is a monotone transform of creatinine and age,
    mainly useful for linear models.
    """
    out = df.copy()
    if {"Kreatynina", "Wiek_rozpoznania"} <= set(out.columns):
        scr_mg_dl = out["Kreatynina"] / 88.4
        out["eGFR_approx"] = (175 * scr_mg_dl ** -1.154 * out["Wiek_rozpoznania"].clip(lower=18) ** -0.203).clip(0, 200)
    for col in ("Kreatynina", "Max_CRP", "Eozynofilia_Krwi_Obwodowej_Wartosc", "Anti-PR3_Wartosc", "Anti-MPO_Wartosc"):
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col])
    manifest = [c for c in out.columns if c.startswith("Manifestacja_")]
    if manifest:
        known = out[manifest].notna().sum(axis=1)
        out["Liczba_Manifestacji_Tak"] = out[manifest].sum(axis=1, min_count=1).where(known > 0)
    if {"Anti-PR3", "Anti-MPO"} <= set(out.columns):
        out["ANCA_PR3_lub_MPO"] = out[["Anti-PR3", "Anti-MPO"]].max(axis=1)
    return out
