"""Tests for registry recoding (src/data/cohort.py) and feature sets."""

import numpy as np
import pandas as pd
import pytest

from src.data.cohort import deduplicate, flag_incomplete_records, load_cohort, recode
from src.data.feature_sets import (
    POST_BASELINE_COLUMNS,
    add_clinical_features,
    baseline_features,
    is_post_baseline,
)


def _raw(**cols):
    n = len(next(iter(cols.values())))
    base = {"Kod": [f"AAAAA{i:04d}" for i in range(n)], "Zgon": [0] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_tri_state_codes_become_yes_no_missing():
    raw = _raw(**{"Manifestacja_Nerki": [1, 2, 0, -1, 3], "Plazmaferezy": [2, 1, 0, 3, -1]})
    out = recode(raw, drop_constant=False)
    assert out["Manifestacja_Nerki"].tolist()[:2] == [1.0, 0.0]
    assert out["Manifestacja_Nerki"].iloc[2:].isna().all()
    assert out["Plazmaferezy"].tolist()[:2] == [0.0, 1.0]
    assert out["Plazmaferezy"].iloc[2:].isna().all()


def test_flag_columns_keep_zero_as_no():
    raw = _raw(**{"Zap_GPA": [0, 1, -1], "Leczenie_Rituximab": [1, 0, -1]})
    out = recode(raw, drop_constant=False)
    assert out["Zap_GPA"].tolist()[:2] == [0.0, 1.0]
    assert np.isnan(out["Zap_GPA"].iloc[2])
    assert out["Leczenie_Rituximab"].tolist()[:2] == [1.0, 0.0]


def test_continuous_sentinels_outliers_and_units():
    raw = _raw(
        Kreatynina=[100.0, 0.0, -1.0, 9016.8],
        Max_CRP=[10.0, 0.0, 5.0, -1.0],
        Eozynofilia_Krwi_Obwodowej_Wartosc=[0, 150, -1, 40],
        Wiek_rozpoznania=[365.25 * 50, 0, 365.25 * 30, -1],
        Opoznienie_Rozpoznia=[30.44 * 2, 10, 31659, 5],
    )
    out = recode(raw, drop_constant=False)
    assert out["Kreatynina"].iloc[0] == 100.0
    assert out["Kreatynina"].iloc[1:].isna().all()
    assert np.isnan(out["Max_CRP"].iloc[1])
    assert np.isnan(out["Eozynofilia_Krwi_Obwodowej_Wartosc"].iloc[0])
    assert out["Wiek_rozpoznania"].iloc[0] == pytest.approx(50)
    assert np.isnan(out["Wiek_rozpoznania"].iloc[1])
    assert out["Opoznienie_Rozpoznia"].iloc[0] == pytest.approx(2)
    # delay is unknown when the age at diagnosis is unknown; implausible delay -> NaN
    assert np.isnan(out["Opoznienie_Rozpoznia"].iloc[1])
    assert np.isnan(out["Opoznienie_Rozpoznia"].iloc[2])


def test_categorical_no_data_codes():
    out = recode(_raw(Pulsy=[0, 4, 5, -1]), drop_constant=False)
    assert out["Pulsy"].tolist()[:2] == [0.0, 4.0]
    assert out["Pulsy"].iloc[2:].isna().all()


def test_single_row_recoding_keeps_columns():
    out = recode(_raw(Manifestacja_Nerki=[1], Kreatynina=[120.0]), drop_constant=False)
    assert {"Manifestacja_Nerki", "Kreatynina"} <= set(out.columns)


def test_incomplete_record_flag():
    manifest = {f"Manifestacja_{i}": [0, 1] for i in range(6)}
    flags = flag_incomplete_records(_raw(**manifest))
    assert flags.tolist() == [True, False]


def test_deduplicate_drops_conflicting_outcomes():
    raw = pd.DataFrame({
        "Kod": ["A", "A", "B", "B", "C", "C"],
        "Zgon": [0, 0, 0, 1, 1, 1],
        "x": [1, 1, 2, 3, 4, 5],
    })
    out, conflicting, notes = deduplicate(raw)
    assert conflicting == ["B"]
    assert sorted(out["Kod"]) == ["A", "C"]
    assert notes


def test_baseline_features_exclude_post_baseline_and_targets():
    columns = ["Plec", "Wiek", "Wiek_rozpoznania", "Leczenie_6m_MMF", "Leczenie_MMF", "Powiklania_Nerki",
               "Zaostrz_Wymagajace_OIT", "Czas_Sterydow", "Dializa", "Zgon", "Przebieg_scalony"]
    for task in ("mortality", "dialysis"):
        kept = baseline_features(columns, task)
        assert kept == ["Plec", "Wiek_rozpoznania", "Leczenie_MMF"]
        assert not any(is_post_baseline(c) for c in kept)
    with pytest.raises(ValueError):
        baseline_features(columns, "other")
    assert "Wiek" in POST_BASELINE_COLUMNS


def test_clinical_features_are_row_wise():
    df = pd.DataFrame({"Kreatynina": [88.4, np.nan], "Wiek_rozpoznania": [50.0, 60.0],
                       "Manifestacja_Nerki": [1.0, np.nan], "Manifestacja_Skora": [0.0, np.nan]})
    out = add_clinical_features(df)
    assert out["eGFR_approx"].iloc[0] > 0 and np.isnan(out["eGFR_approx"].iloc[1])
    assert out["Liczba_Manifestacji_Tak"].iloc[0] == 1 and np.isnan(out["Liczba_Manifestacji_Tak"].iloc[1])


def test_real_cohort_has_no_leaky_columns():
    cohort = load_cohort()
    cols = baseline_features(cohort.data.columns, "mortality")
    assert "Zgon" in cohort.data.columns and "Zgon" not in cols
    assert not any(c.startswith(("Powiklania_", "Leczenie_6m_", "Zaostrz_")) for c in cols)
    # code 0 must not survive in tri-state columns
    assert set(cohort.data["Manifestacja_Nerki"].dropna().unique()) <= {0.0, 1.0}
    assert cohort.data["Kod" if "Kod" in cohort.data else "Zgon"].notna().all()
