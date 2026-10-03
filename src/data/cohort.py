"""
Cohort loading and semantic recoding of the raw vasculitis registry.

The raw CSV mixes several coding schemes and uses sentinel values for missing
data. Treating those sentinels as numbers leaks the outcome: a block of ~94
patients (mostly three centres) has code 0 in most clinical fields and an 85%
mortality rate, so "field is 0" becomes a proxy for death.

Recoding rules (verified against value distributions in data/raw):

* FLAG columns (0 = no, 1 = yes, -1 = missing): Zap_*, Leczenie_*, ANCA_*, Biops_*, ...
* TRI-STATE columns (1 = yes, 2 = no, 3 = unknown/not tested, 0 / -1 = missing):
  Manifestacja_*, Zaostrz_*, Plazmaferezy, Powiklania_*, Anti-PR3, Dializa, ...
  -> recoded to 1 / 0 / NaN.
* CATEGORICAL columns keep their codes, -1 (and documented "no data" codes) -> NaN.
* CONTINUOUS columns: -1 -> NaN, physiologically impossible values -> NaN,
  ages in days -> years, diagnostic delay in days -> months.

Nothing here is fitted on data, so it is safe to apply before splitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "aktualne_dane.csv"

ID_COLUMN = "Kod"
MORTALITY_TARGET = "Zgon"
DIALYSIS_TARGET = "Dializa"

DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.44

TRI_STATE_PREFIXES = ("Manifestacja_", "Zaostrz_", "Powiklania_", "Powiklanie_")
TRI_STATE_COLUMNS = {
    "Plazmaferezy",
    "Dializa",
    "Anti-PR3",
    "Anti-MPO",
    "Anti-GBM",
    "Krioglobuliny",
    "Eozynofilia_Krwi_Obwodowej",
}

FLAG_PREFIXES = ("Zap_", "Chor_", "Zes_", "Zesp_", "Leczenie_", "ANCA", "Biops_", "Objawy_")
FLAG_COLUMNS = {
    "Plec",
    "Inne_Rozpoznania",
    "Cukrzyca",
    "Nadcisnienie",
    "Histologia",
    "Laboratorium",
    "Inne_Rozpoznanie",
}

# Categorical codes; extra per-column codes that mean "no data".
CATEGORICAL_COLUMNS: Dict[str, set] = {
    "Rozpoznanie": set(),
    "Praca": set(),
    "Palenie_Tytoniu": set(),
    "Pulsy": {5},  # code 5 occurs only in the incomplete-record block
    "Droga_Podania": set(),
    "Biopsja_Wynik": set(),
    "Przebieg_scalony": {9},
}

# Upper plausibility limits for continuous variables (values above -> NaN).
PLAUSIBLE_MAX = {
    "Kreatynina": 3000.0,  # umol/L; 8486 / 9017 are unit / entry errors
    "Opoznienie_Rozpoznia": 20000.0,  # days; larger values look like an age typed in
    "Liczba_Zaostrzen": 50.0,
    "Sterydy_Dawka_g": 200.0,
    "Sterydy_Dawka_mg": 5000.0,
}
# Zero is not a real value here (min. real eosinophil count is 10; CRP 0 only in incomplete records).
ZERO_IS_MISSING = {"Kreatynina", "Wiek_rozpoznania", "Eozynofilia_Krwi_Obwodowej_Wartosc", "Max_CRP"}

MANIFESTATION_PREFIX = "Manifestacja_"
INCOMPLETE_RECORD_MIN_ZEROS = 5


@dataclass
class Cohort:
    """Recoded cohort plus bookkeeping needed for the audit report."""

    data: pd.DataFrame
    centre: pd.Series
    incomplete_record: pd.Series
    excluded_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def is_tri_state(column: str) -> bool:
    return column in TRI_STATE_COLUMNS or column.startswith(TRI_STATE_PREFIXES)


def is_flag(column: str) -> bool:
    return column in FLAG_COLUMNS or column.startswith(FLAG_PREFIXES)


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the pipe-separated CSV with stripped column names."""
    df = pd.read_csv(path, sep="|")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def flag_incomplete_records(raw: pd.DataFrame) -> pd.Series:
    """Rows with code 0 in many manifestation fields (incomplete registry entries)."""
    manifest = [c for c in raw.columns if c.startswith(MANIFESTATION_PREFIX)]
    return (raw[manifest] == 0).sum(axis=1) >= INCOMPLETE_RECORD_MIN_ZEROS


def recode(raw: pd.DataFrame, *, drop_constant: bool = True) -> pd.DataFrame:
    """Apply semantic recoding; returns a numeric frame (ID column dropped).

    drop_constant=False keeps all columns (needed when recoding single rows at
    prediction time).
    """
    cols: Dict[str, pd.Series] = {}
    for column in raw.columns:
        if column == ID_COLUMN:
            continue
        s = pd.to_numeric(raw[column], errors="coerce").astype(float)
        s = s.mask(s == -1)

        if column == MORTALITY_TARGET:
            pass
        elif is_tri_state(column):
            s = s.map({1.0: 1.0, 2.0: 0.0})  # 0 / 3 / anything else -> NaN
        elif is_flag(column):
            s = s.where(s.isin([0.0, 1.0]))
        elif column in CATEGORICAL_COLUMNS:
            s = s.mask(s.isin(CATEGORICAL_COLUMNS[column]))
        else:
            if column in ZERO_IS_MISSING:
                s = s.mask(s == 0)
            if column in PLAUSIBLE_MAX:
                s = s.mask(s > PLAUSIBLE_MAX[column])
            s = s.mask(s < 0)
        cols[column] = s
    out = pd.DataFrame(cols, index=raw.index)

    for column in ("Wiek", "Wiek_rozpoznania"):
        if column in out:
            out[column] = out[column] / DAYS_PER_YEAR
    if "Opoznienie_Rozpoznia" in out:
        delay = out["Opoznienie_Rozpoznia"]
        if "Wiek_rozpoznania" in out:
            delay = delay.mask(out["Wiek_rozpoznania"].isna())
        out["Opoznienie_Rozpoznia"] = delay / DAYS_PER_MONTH

    if "Liczba_Zajetych_Narzadow" in out:
        manifest = [c for c in out.columns if c.startswith(MANIFESTATION_PREFIX)]
        mostly_missing = out[manifest].isna().sum(axis=1) >= INCOMPLETE_RECORD_MIN_ZEROS
        out["Liczba_Zajetych_Narzadow"] = out["Liczba_Zajetych_Narzadow"].mask(mostly_missing)

    if not drop_constant:
        return out
    constant = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
    return out.drop(columns=constant)


def deduplicate(raw: pd.DataFrame) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Drop exact duplicates, keep first of near-duplicates, drop IDs with conflicting outcome."""
    notes: List[str] = []
    n0 = len(raw)
    raw = raw.drop_duplicates()
    if len(raw) < n0:
        notes.append(f"dropped {n0 - len(raw)} exact duplicate row(s)")

    dup = raw[raw[ID_COLUMN].duplicated(keep=False)]
    conflicting = (
        dup.groupby(ID_COLUMN)[MORTALITY_TARGET].nunique().loc[lambda s: s > 1].index.tolist()
    )
    if conflicting:
        raw = raw[~raw[ID_COLUMN].isin(conflicting)]
        notes.append(f"excluded IDs with conflicting {MORTALITY_TARGET}: {conflicting}")

    n1 = len(raw)
    raw = raw.drop_duplicates(subset=[ID_COLUMN], keep="first")
    if len(raw) < n1:
        notes.append(f"kept first of {n1 - len(raw)} near-duplicate ID row(s)")
    return raw.reset_index(drop=True), conflicting, notes


def load_cohort(path: str | Path = DEFAULT_DATA_PATH) -> Cohort:
    """Load, deduplicate and recode the registry."""
    raw = load_raw(path)
    raw, conflicting, notes = deduplicate(raw)
    data = recode(raw)
    return Cohort(
        data=data,
        centre=raw[ID_COLUMN].astype(str).str[:5].rename("centre"),
        incomplete_record=flag_incomplete_records(raw).rename("incomplete_record"),
        excluded_ids=conflicting,
        notes=notes,
    )
