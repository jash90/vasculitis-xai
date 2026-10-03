#!/usr/bin/env python3
"""
Step 1-3 of the model-improvement protocol: baseline + leakage ablation.

1. Re-scores the CURRENT artifacts (backup of models/saved) on the original
   hold-out split (seed 42, 80/20) - numbers are recomputed, not read from JSON.
2. Ablation under repeated stratified CV (5x5); each row adds one fix:
     A  legacy: all columns, only -1 treated as missing, codes 0/1/2 as numbers
     B  + semantic recoding (code 0 / 3 = missing, 1/2 -> yes/no, outliers)
     C  + post-baseline features removed            (= honest baseline)
     D  C evaluated without the 94 incomplete records (sensitivity)
     E  C under leave-one-centre-out                 (transportability)
   plus a majority-class reference.
3. Writes reports/model_improvement/{holdout_current_artifacts,ablation,baseline}.csv
   and OOF predictions (npz) used later for DeLong comparisons.

Usage: venv/bin/python scripts/model_audit.py [--repeats 5] [--backup-dir models/_backup_2026-10-01]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import ID_COLUMN, MORTALITY_TARGET, deduplicate, load_cohort, load_raw  # noqa: E402
from src.data.feature_sets import baseline_features  # noqa: E402
from src.models.evaluation_protocol import binary_metrics, bootstrap_ci, grouped_cv, repeated_cv  # noqa: E402
from src.models.model_zoo import make_factory  # noqa: E402

warnings.filterwarnings("ignore")
REPORT_DIR = PROJECT_ROOT / "reports" / "model_improvement"
ABLATION_MODELS = ("logreg", "xgboost", "catboost")

LEGACY_BINARY = {
    "Manifestacja_Miesno-Szkiel", "Manifestacja_Skora", "Manifestacja_Wzrok",
    "Manifestacja_Sercowo-Naczyniowy", "Manifestacja_Pokarmowy", "Manifestacja_Nerki",
    "Manifestacja_Moczowo-Plciowy", "Manifestacja_Zajecie_CSN", "Manifestacja_Neurologiczny",
    "Zaostrz_Wymagajace_Hospital", "Zaostrz_Wymagajace_OIT", "Pulsy", "Plazmaferezy", "Biopsja_Wynik",
}


def legacy_aligned_matrix(raw: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Exact preprocessing of the previous scripts/retrain_aligned_models.py."""
    s_all = raw[feature_names].apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    clean = pd.DataFrame(index=raw.index)
    for name in feature_names:
        s = s_all[name]
        if name in LEGACY_BINARY:
            clean[name] = (s.fillna(0) > 0).astype(float)
        elif name == "Wiek_rozpoznania":
            clean[name] = (s / 365.25).clip(0, 120).fillna(0.0)
        elif name == "Opoznienie_Rozpoznia":
            clean[name] = (s / 30.44).clip(lower=0).fillna(0.0)
        elif name == "Liczba_Zajetych_Narzadow":
            clean[name] = s.clip(0, 20).fillna(0.0)
        else:
            clean[name] = s.clip(lower=0).fillna(0.0)
    return clean.to_numpy(dtype=np.float64)


def score_current_artifacts(backup_dir: Path) -> pd.DataFrame:
    raw = load_raw()
    y = raw[MORTALITY_TARGET].astype(int).to_numpy()
    idx_train, idx_test = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=42, stratify=y)
    rows = []

    names = json.loads((backup_dir / "feature_names.json").read_text())
    X = legacy_aligned_matrix(raw, names)
    for key, fname in {"xgboost": "best_model.joblib", "random_forest": "random_forest_model.joblib",
                       "lightgbm": "lightgbm_model.joblib"}.items():
        model = joblib.load(backup_dir / fname)
        p = model.predict_proba(X[idx_test])[:, 1]
        lo, hi = bootstrap_ci(y[idx_test], p)
        rows.append({"artifact": "aligned_20", "model": key, **binary_metrics(y[idx_test], p, 0.5),
                     "roc_auc_ci_low": lo, "roc_auc_ci_high": hi})

    for sub in ("auc", "auc_all"):
        d = backup_dir / sub
        meta = json.loads((d / "auc_metadata.json").read_text())
        Xa = raw.drop(columns=[MORTALITY_TARGET, ID_COLUMN]).apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
        Xa = Xa[meta["raw_feature_names"]].iloc[idx_test]
        Z = joblib.load(d / "auc_scaler.joblib").transform(
            joblib.load(d / "auc_selector.joblib").transform(joblib.load(d / "auc_imputer.joblib").transform(Xa)))
        for row in json.loads((d / "auc_model_comparison.json").read_text()):
            model = joblib.load(d / f"{row['model']}_model.joblib")
            p = model.predict_proba(Z)[:, 1]
            lo, hi = bootstrap_ci(y[idx_test], p)
            rows.append({"artifact": f"{sub}_{meta['n_features']}", "model": row["model"],
                         **binary_metrics(y[idx_test], p, 0.5), "roc_auc_ci_low": lo, "roc_auc_ci_high": hi,
                         "selected_on_test_as_best": row["model"] == meta["best_model"]})
    return pd.DataFrame(rows)


def legacy_frame() -> tuple[pd.DataFrame, np.ndarray]:
    raw, _, _ = deduplicate(load_raw())
    y = raw[MORTALITY_TARGET].astype(int).to_numpy()
    X = raw.drop(columns=[MORTALITY_TARGET, ID_COLUMN]).apply(pd.to_numeric, errors="coerce").replace(-1, np.nan)
    return X.astype(float), y


def run_ablation(n_repeats: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    cohort = load_cohort()
    data = cohort.data
    y = data[MORTALITY_TARGET].astype(int).to_numpy()
    complete = ~cohort.incomplete_record.to_numpy()

    X_legacy, y_legacy = legacy_frame()
    X_recoded = data.drop(columns=[MORTALITY_TARGET])
    base_cols = baseline_features(data.columns, "mortality")
    X_base = data[base_cols]

    variants = {
        "A_legacy_all_columns": (X_legacy, y_legacy),
        "B_recoded_all_columns": (X_recoded, y),
        "C_baseline_features": (X_base, y),
        "D_baseline_without_incomplete": (X_base[complete], y[complete]),
    }
    rows, oof_store = [], {}
    for variant, (X, yy) in variants.items():
        for kind in ABLATION_MODELS:
            res = repeated_cv(make_factory(kind, X.columns), X, yy, name=kind, n_repeats=n_repeats)
            rows.append({"variant": variant, "n": len(yy), "n_features": X.shape[1], **res.summary()})
            oof_store[f"{variant}__{kind}"] = res.oof
            print(f"  {variant:32s} {kind:10s} AUC={rows[-1]['roc_auc_mean']:.3f}±{rows[-1]['roc_auc_sd']:.3f}")
        rows.append({"variant": variant, "model": "majority_class", "n": len(yy), "roc_auc_mean": 0.5,
                     "pr_auc_mean": float(yy.mean()), "brier_mean": float(yy.mean() * (1 - yy.mean()))})

    loco_rows = []
    for kind in ABLATION_MODELS:
        res = grouped_cv(make_factory(kind, base_cols), X_base, y, cohort.centre, name=kind)
        rows.append({"variant": "E_baseline_leave_one_centre_out", "model": kind, "n": len(y),
                     "n_features": len(base_cols), "roc_auc_mean": res["pooled_roc_auc"],
                     **{f"{k}_mean": v for k, v in res["pooled"].items() if k != "roc_auc"}})
        loco_rows.append(res["per_centre"].assign(model=kind))
        print(f"  {'E_baseline_LOCO':32s} {kind:10s} AUC={res['pooled_roc_auc']:.3f}")
    oof_store["y_cohort"] = y
    oof_store["y_legacy"] = y_legacy
    oof_store["complete_mask"] = complete
    return pd.DataFrame(rows), oof_store, pd.concat(loco_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backup-dir", type=Path, default=PROJECT_ROOT / "models" / "_backup_2026-10-01")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    (REPORT_DIR / "cohort_notes.json").write_text(json.dumps({
        "n_after_cleaning": int(len(cohort.data)),
        "deaths": int(cohort.data[MORTALITY_TARGET].sum()),
        "incomplete_records": int(cohort.incomplete_record.sum()),
        "incomplete_records_mortality": float(cohort.data[MORTALITY_TARGET][cohort.incomplete_record].mean()),
        "notes": cohort.notes,
        "baseline_features": baseline_features(cohort.data.columns, "mortality"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("1) Current artifacts on the original hold-out")
    holdout = score_current_artifacts(args.backup_dir)
    holdout.to_csv(REPORT_DIR / "holdout_current_artifacts.csv", index=False)
    print(holdout[["artifact", "model", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "brier"]]
          .round(3).to_string(index=False))

    print(f"\n2) Ablation ({args.repeats}x5 CV)")
    ablation, oof, loco = run_ablation(args.repeats)
    ablation.to_csv(REPORT_DIR / "ablation.csv", index=False)
    loco.to_csv(REPORT_DIR / "loco_per_centre.csv", index=False)
    ablation[ablation["variant"] == "C_baseline_features"].to_csv(REPORT_DIR / "baseline.csv", index=False)
    np.savez_compressed(REPORT_DIR / "oof_audit.npz", **oof)
    print(f"\nSaved to {REPORT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
