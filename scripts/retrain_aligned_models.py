#!/usr/bin/env python3
"""
Retrain the three API models (/predict, /predict/all, XAI) on the 20 form features.

Changes against the previous version (see reports/model_improvement/REPORT.md):
  * data come from src.data.cohort: 1 = yes / 2 = no are no longer merged into
    one value, code 0 / -1 / 3 means "missing", implausible values -> missing;
  * the three follow-up features (Zaostrz_Wymagajace_Hospital,
    Zaostrz_Wymagajace_OIT, Czas_Sterydow) are replaced by features known at
    diagnosis: Manifestacja_Oddechowy, Manifestacja_Nos/Ucho/Gardlo, Max_CRP
    (chosen on the development split only, by 5x5 CV gain of a logistic model);
  * missing values are imputed with training medians INSIDE the model
    (Pipeline), so the API can pass NaN for unknown fields;
  * hyperparameters are tuned with Optuna on the development split, the choice
    calibrated / uncalibrated is made by repeated CV (Brier), and the 20%
    hold-out is scored once and only reported.

Artifacts keep the API contract: best_model.joblib (xgboost key),
random_forest_model.joblib, lightgbm_model.joblib, feature_names.json,
X_train / X_test / X_background / y_* (20 columns, NaN imputed with training
medians so LIME/SHAP backgrounds stay finite) and feature_defaults.json.

Additionally (UI survival view): a Random Survival Forest on the same 20 features
(time = Wiek - Wiek_rozpoznania, event = Zgon) -> survival_rsf.joblib, and
models_metadata.json with the measured metrics that /model/info serves.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import MORTALITY_TARGET, load_cohort  # noqa: E402
from src.models.evaluation_protocol import binary_metrics, bootstrap_ci, grouped_cv, repeated_cv  # noqa: E402

warnings.filterwarnings("ignore")
OUT_DIR = PROJECT_ROOT / "models" / "saved"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_improvement" / "serving"

# Order = feature_names.json = patient_to_array (src/api/schemas.py)
FEATURE_NAMES = [
    "Wiek_rozpoznania",
    "Opoznienie_Rozpoznia",
    "Manifestacja_Miesno-Szkiel",
    "Manifestacja_Skora",
    "Manifestacja_Wzrok",
    "Manifestacja_Sercowo-Naczyniowy",
    "Manifestacja_Pokarmowy",
    "Manifestacja_Nerki",
    "Manifestacja_Moczowo-Plciowy",
    "Manifestacja_Zajecie_CSN",
    "Manifestacja_Neurologiczny",
    "Manifestacja_Oddechowy",
    "Manifestacja_Nos/Ucho/Gardlo",
    "Liczba_Zajetych_Narzadow",
    "Kreatynina",
    "Max_CRP",
    "Pulsy",
    "Plazmaferezy",
    "Eozynofilia_Krwi_Obwodowej_Wartosc",
    "Biopsja_Wynik",
]
SEARCH_TRIALS = 30


def form_space(data: pd.DataFrame) -> pd.DataFrame:
    """Recoded cohort -> units/encoding of the API form (PatientInput)."""
    X = data[FEATURE_NAMES].copy()
    # categorical codes in the registry; the form asks yes/no
    X["Pulsy"] = (X["Pulsy"] > 0).astype(float).where(X["Pulsy"].notna())
    X["Biopsja_Wynik"] = (X["Biopsja_Wynik"] > 0).astype(float).where(X["Biopsja_Wynik"].notna())
    return X.astype(np.float64)


def make_estimator(kind: str, params: dict | None = None):
    params = dict(params or {})
    if kind == "xgboost":
        from xgboost import XGBClassifier

        base = dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0)
        base.update(params)
        est = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=2, **base)
    elif kind == "random_forest":
        base = dict(n_estimators=500, min_samples_leaf=5, max_features="sqrt")
        base.update(params)
        est = RandomForestClassifier(random_state=42, n_jobs=2, **base)
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier

        base = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)
        base.update(params)
        est = LGBMClassifier(random_state=42, n_jobs=2, verbose=-1, **base)
    else:
        raise ValueError(kind)
    # positional imputer: the API passes plain ndarrays (and DataFrames for DALEX)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])


def tune(kind: str, X: np.ndarray, y: np.ndarray) -> dict:
    import optuna

    from src.models.tuning import SEARCH_SPACES

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(3, shuffle=True, random_state=42)

    def objective(trial):
        model = make_estimator(kind, SEARCH_SPACES[kind](trial))
        return cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=SEARCH_TRIALS)
    return SEARCH_SPACES[kind](optuna.trial.FixedTrial(study.best_params))


def make_survival_model():
    from sksurv.ensemble import RandomSurvivalForest

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomSurvivalForest(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                       n_jobs=2, random_state=42)),
    ])


def train_survival(cohort, X: pd.DataFrame, idx_dev: np.ndarray, idx_hold: np.ndarray) -> dict:
    """RSF on the form features; returns metrics (5x3 CV on dev, hold-out once)."""
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
    from sksurv.util import Surv

    from src.models.survival import EVAL_TIMES, TAU, sksurv_fp, survival_cv, truncate

    data = cohort.data
    fu = (data["Wiek"] - data["Wiek_rozpoznania"]).to_numpy()
    known = ~np.isnan(fu)
    dev = np.intersect1d(idx_dev, np.flatnonzero(known))
    hold = np.intersect1d(idx_hold, np.flatnonzero(known))

    def surv(idx):
        return Surv.from_arrays(data[MORTALITY_TARGET].to_numpy()[idx].astype(bool), np.clip(fu[idx], 1 / 365.25, None))

    y_dev, y_hold = surv(dev), surv(hold)
    X_dev, X_hold = X.iloc[dev].reset_index(drop=True), X.iloc[hold].reset_index(drop=True)
    cv = survival_cv("rsf_form20", sksurv_fp(make_survival_model), X_dev, y_dev, n_repeats=3, verbose=False)["summary"]
    model = make_survival_model().fit(X_dev.to_numpy(), y_dev)
    risk = model.predict(X_hold.to_numpy())
    yd_e, yh_e = truncate(y_dev), truncate(y_hold)
    times = EVAL_TIMES[EVAL_TIMES < yh_e["time"].max()]
    auc_t, _ = cumulative_dynamic_auc(yd_e, yh_e, risk, times)
    joblib.dump(model, OUT_DIR / "survival_rsf.joblib", compress=3)
    hold_m = {"harrell_c": float(concordance_index_censored(y_hold["event"], y_hold["time"], risk)[0]),
              "uno_c": float(concordance_index_ipcw(yd_e, yh_e, risk, tau=TAU)[0]),
              **{f"auc_{t:g}y": float(a) for t, a in zip(times, auc_t)}}
    print(f"  [survival RSF ] CV C={cv['harrell_c_mean']:.3f}±{cv['harrell_c_sd']:.3f} AUC(t) "
          f"{cv.get('auc_1y_mean', np.nan):.3f}/{cv.get('auc_3y_mean', np.nan):.3f}/{cv.get('auc_5y_mean', np.nan):.3f} | "
          f"hold-out C={hold_m['harrell_c']:.3f}")
    return {
        "model": "Random Survival Forest", "n_train": int(len(dev)), "deaths_train": int(y_dev["event"].sum()),
        "median_follow_up_years": float(np.median(y_dev["time"])),
        "cv": {k: float(v) for k, v in cv.items() if k != "model"}, "holdout": hold_m,
        "horizons_years": [1, 3, 5],
    }


def main() -> int:
    cohort = load_cohort()
    data = cohort.data
    y = data[MORTALITY_TARGET].astype(int).to_numpy()
    X = form_space(data)
    idx_dev, idx_hold = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    X_dev, X_hold, y_dev, y_hold = X.iloc[idx_dev], X.iloc[idx_hold], y[idx_dev], y[idx_hold]
    print(f"n={len(y)} dev={len(y_dev)} hold-out={len(y_hold)} deaths={y.sum()} features={len(FEATURE_NAMES)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"feature_names": FEATURE_NAMES, "models": {}}

    for fname, kind in {"best_model.joblib": "xgboost", "random_forest_model.joblib": "random_forest",
                        "lightgbm_model.joblib": "lightgbm"}.items():
        params = tune(kind, X_dev.to_numpy(), y_dev)
        variants = {
            "uncalibrated": lambda: make_estimator(kind, params),
            "sigmoid": lambda: CalibratedClassifierCV(make_estimator(kind, params), method="sigmoid", cv=5),
        }
        cv_rows = {name: repeated_cv(f, X_dev.reset_index(drop=True), y_dev, name=name, n_repeats=5,
                                     sensitivity_target=None).summary() for name, f in variants.items()}
        choice = min(cv_rows, key=lambda k: cv_rows[k]["brier_mean"])
        model = variants[choice]().fit(X_dev.to_numpy(), y_dev)
        p_hold = model.predict_proba(X_hold.to_numpy())[:, 1]
        lo, hi = bootstrap_ci(y_hold, p_hold)
        hold = {**binary_metrics(y_hold, p_hold, 0.5), "roc_auc_ci_low": lo, "roc_auc_ci_high": hi}
        joblib.dump(model, OUT_DIR / fname)
        report["models"][kind] = {"file": fname, "params": params, "calibration": choice,
                                  "dev_cv": cv_rows[choice], "dev_cv_alternatives": cv_rows, "holdout": hold}
        print(f"  [{kind:13s}] calib={choice:12s} CV AUC={cv_rows[choice]['roc_auc_mean']:.3f}"
              f"±{cv_rows[choice]['roc_auc_sd']:.3f} Brier={cv_rows[choice]['brier_mean']:.3f} | "
              f"hold-out AUC={hold['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}]")

    # Transportability check for the served primary model (leave-one-centre-out on dev)
    dev_centres = cohort.centre.iloc[idx_dev].reset_index(drop=True)
    loco = grouped_cv(lambda: make_estimator("xgboost", report["models"]["xgboost"]["params"]),
                      X_dev.reset_index(drop=True), y_dev, dev_centres)
    report["models"]["xgboost"]["leave_one_centre_out_auc"] = loco["pooled_roc_auc"]
    print(f"  [xgboost LOCO ] pooled AUC={loco['pooled_roc_auc']:.3f}")

    survival = train_survival(cohort, X, idx_dev, idx_hold)

    # XAI data: same 20-column raw space, NaN -> training medians
    medians = X_dev.median()
    X_train_xai = X_dev.fillna(medians).to_numpy()
    X_test_xai = X_hold.fillna(medians).to_numpy()
    bg_idx = np.random.RandomState(42).choice(len(X_train_xai), size=min(100, len(X_train_xai)), replace=False)
    joblib.dump(X_train_xai, OUT_DIR / "X_train.joblib")
    joblib.dump(X_test_xai, OUT_DIR / "X_test.joblib")
    joblib.dump(X_train_xai[bg_idx], OUT_DIR / "X_background.joblib")
    joblib.dump(y_dev, OUT_DIR / "y_train.joblib")
    joblib.dump(y_hold, OUT_DIR / "y_test.joblib")
    joblib.dump(y_dev[bg_idx], OUT_DIR / "y_background.joblib")
    (OUT_DIR / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    (OUT_DIR / "feature_defaults.json").write_text(json.dumps(medians.round(4).to_dict(), ensure_ascii=False,
                                                              indent=2), encoding="utf-8")
    labels = {"xgboost": "XGBoost", "random_forest": "Random Forest", "lightgbm": "LightGBM"}
    metadata = {
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "task": "Śmiertelność (zgon w okresie obserwacji), predykcja w chwili rozpoznania",
        "n_patients": int(len(y)), "n_deaths": int(y.sum()),
        "n_train": int(len(y_dev)), "n_holdout": int(len(y_hold)),
        "feature_names": FEATURE_NAMES,
        "classifiers": {
            k: {"label": labels[k], "calibration": v["calibration"],
                "cv": {m: v["dev_cv"].get(f"{m}_mean") for m in ("roc_auc", "pr_auc", "brier", "cal_slope")}
                | {"roc_auc_sd": v["dev_cv"]["roc_auc_sd"], "roc_auc_ci": [v["dev_cv"]["roc_auc_ci_low"],
                                                                          v["dev_cv"]["roc_auc_ci_high"]]},
                "holdout": {m: v["holdout"][m] for m in ("roc_auc", "pr_auc", "brier")}
                | {"roc_auc_ci": [v["holdout"]["roc_auc_ci_low"], v["holdout"]["roc_auc_ci_high"]]},
                **({"leave_one_centre_out_auc": v["leave_one_centre_out_auc"]} if "leave_one_centre_out_auc" in v else {})}
            for k, v in report["models"].items()
        },
        "survival": survival,
        "limitations": [
            "Dane z jednego rejestru (893 pacjentów, 187 zgonów); brak walidacji zewnętrznej.",
            "Przy walidacji na niewidzianym ośrodku dyskryminacja spada (AUC ≈ 0.75).",
            "92 niekompletne rekordy (głównie zmarli) mogą zawyżać wyniki.",
            "Czas obserwacji wyliczony z wieku przy ostatnim kontakcie/zgonie (do potwierdzenia).",
            "Narzędzie badawcze — nie zastępuje oceny klinicznej.",
        ],
    }
    (OUT_DIR / "models_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float),
                                                  encoding="utf-8")
    (REPORT_DIR / "aligned_models.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    print(f"Saved models + XAI data to {OUT_DIR}, report to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
