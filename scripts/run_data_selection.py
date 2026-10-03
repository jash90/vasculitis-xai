#!/usr/bin/env python3
"""
Data selection experiments: which features and which patients give the best,
still defensible, mortality model.

Two models are evaluated with the same protocol:
  * binary XGBoost (the model chosen in REPORT.md section 3; target Zgon)
  * Random Survival Forest (REPORT.md section 7; time = Wiek - Wiek_rozpoznania)

A. Feature selection (development split, repeated 5-fold CV):
   - all at-diagnosis features (reference)
   - top-k by mutual information, fitted INSIDE each training fold
   - stability selection (bootstrap L1), fitted inside each training fold
   - drop-one-group ablation over clinical feature groups
   - compact clinical set (FFS-like) and the 20 API form features
   Selection rule (fixed in advance): highest mean metric, then among
   configurations within one SD the one with the fewest features.

B. Patient selection (criteria that do not look at the outcome):
   - incomplete registry records excluded
   - rows with > 20% missing at-diagnosis features excluded
   - only centres with >= 20 patients
   Each population is evaluated within itself (results describe that population).

C. Hold-out (20%, same split as before) scored once for the selected feature
   configuration vs all features, on all hold-out patients.

Usage: venv/bin/python scripts/run_data_selection.py [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import MORTALITY_TARGET, load_cohort  # noqa: E402
from src.data.feature_sets import baseline_features  # noqa: E402
from src.models.evaluation_protocol import bootstrap_ci, delong_test, repeated_cv  # noqa: E402
from src.models.model_zoo import make_factory  # noqa: E402
from src.models.selectors import StabilitySelector, TopKMutualInfo  # noqa: E402
from src.models.survival import (  # noqa: E402
    EVAL_TIMES, TAU, paired_c_bootstrap, sksurv_fp, survival_cv, survival_factory, truncate,
)

warnings.filterwarnings("ignore")
OUT = PROJECT_ROOT / "reports" / "model_improvement" / "data_selection"
TOP_K = (5, 10, 15, 20, 30, 45)

GROUPS = {
    "demografia_i_wywiad": lambda c: c in {"Plec", "Wiek_rozpoznania", "Opoznienie_Rozpoznia", "Praca",
                                           "Palenie_Tytoniu", "Paczkolata", "Cukrzyca", "Nadcisnienie"},
    "typ_zapalenia": lambda c: c.startswith(("Zap_", "Zes_", "Chor_")),
    "podstawa_rozpoznania": lambda c: c in {"Rozpoznanie", "Histologia", "Laboratorium", "Inne_Rozpoznanie"}
    or c.startswith("Objawy_"),
    "manifestacje_narzadowe": lambda c: c.startswith("Manifestacja_") or c == "Liczba_Zajetych_Narzadow",
    "laboratoria": lambda c: c in {"Kreatynina", "Max_CRP", "Eozynofilia_Krwi_Obwodowej_Wartosc",
                                   "Eozynofilia_Krwi_Obwodowej"},
    "serologia": lambda c: c.startswith(("ANCA", "Anti-")) or c == "Krioglobuliny",
    "biopsja": lambda c: c.startswith("Biops"),
    "leczenie_indukcyjne": lambda c: (c.startswith("Leczenie_") and not c.startswith("Leczenie_6m_"))
    or c in {"Pulsy", "Droga_Podania", "Plazmaferezy"},
}
COMPACT_CLINICAL = [  # FFS-2009-like set + age, renal function, inflammation, subtype
    "Wiek_rozpoznania", "Kreatynina", "Max_CRP", "Eozynofilia_Krwi_Obwodowej_Wartosc",
    "Manifestacja_Sercowo-Naczyniowy", "Manifestacja_Pokarmowy", "Manifestacja_Nos/Ucho/Gardlo",
    "Manifestacja_Nerki", "Manifestacja_Zajecie_CSN", "Manifestacja_Oddechowy", "Liczba_Zajetych_Narzadow",
    "Zap_GPA", "Zap_MPA", "Plec",
]
API_FORM = [
    "Wiek_rozpoznania", "Opoznienie_Rozpoznia", "Manifestacja_Miesno-Szkiel", "Manifestacja_Skora",
    "Manifestacja_Wzrok", "Manifestacja_Sercowo-Naczyniowy", "Manifestacja_Pokarmowy", "Manifestacja_Nerki",
    "Manifestacja_Moczowo-Plciowy", "Manifestacja_Zajecie_CSN", "Manifestacja_Neurologiczny",
    "Manifestacja_Oddechowy", "Manifestacja_Nos/Ucho/Gardlo", "Liczba_Zajetych_Narzadow", "Kreatynina",
    "Max_CRP", "Pulsy", "Plazmaferezy", "Eozynofilia_Krwi_Obwodowej_Wartosc", "Biopsja_Wynik",
]


def feature_configs(base: List[str]) -> Dict[str, Dict[str, Any]]:
    cfgs: Dict[str, Dict[str, Any]] = {"all_features": {"cols": base, "selector": None, "n": len(base)}}
    for k in TOP_K:
        cfgs[f"top{k}_mutual_info"] = {"cols": base, "selector": (lambda k=k: TopKMutualInfo(k)), "n": k}
    cfgs["stability_selection"] = {"cols": base, "selector": lambda: StabilitySelector(), "n": None}
    for g, rule in GROUPS.items():
        cols = [c for c in base if not rule(c)]
        cfgs[f"without_{g}"] = {"cols": cols, "selector": None, "n": len(cols)}
    cfgs["compact_clinical"] = {"cols": [c for c in COMPACT_CLINICAL if c in base], "selector": None,
                                "n": len([c for c in COMPACT_CLINICAL if c in base])}
    cfgs["api_form_20"] = {"cols": [c for c in API_FORM if c in base], "selector": None,
                           "n": len([c for c in API_FORM if c in base])}
    return cfgs


def n_selected(cfg: Dict[str, Any]) -> int:
    """Expanded number of input features (one-hot not counted) used by a configuration."""
    if cfg["n"] is not None:
        return cfg["n"]
    return 10**6  # unknown in advance (stability selection) -> treated as least simple


def binary_factory(cfg: Dict[str, Any]) -> Callable:
    return make_factory("xgboost", cfg["cols"], balance="none", selector=cfg["selector"])


def rsf_fp(cfg: Dict[str, Any]) -> Callable:
    return sksurv_fp(survival_factory("rsf", cfg["cols"], selector=cfg["selector"]))


def one_sd_rule(rows: pd.DataFrame, metric: str) -> str:
    best = rows.loc[rows[f"{metric}_mean"].idxmax()]
    sd = 0.0 if pd.isna(best[f"{metric}_sd"]) else best[f"{metric}_sd"]
    eligible = rows[rows[f"{metric}_mean"] >= best[f"{metric}_mean"] - sd]
    return eligible.sort_values(["n_inputs", f"{metric}_mean"], ascending=[True, False]).iloc[0]["config"]


def stability_report(X: pd.DataFrame, y, cols: List[str]) -> pd.DataFrame:
    """Selection frequencies on the whole development set (for interpretation only)."""
    from src.models.model_zoo import make_preprocessor

    prep = make_preprocessor(cols, scale=True).fit(X[cols], y)
    names = list(prep.get_feature_names_out())
    sel = StabilitySelector().fit(prep.transform(X[cols]), y)
    return pd.DataFrame({"feature": names, "frequency": sel.frequency_}).sort_values("frequency", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cohort = load_cohort()
    data = cohort.data.copy()
    data["fu_years"] = data["Wiek"] - data["Wiek_rozpoznania"]
    y_all = data[MORTALITY_TARGET].astype(int).to_numpy()
    idx_dev, idx_hold = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42, stratify=y_all)
    in_dev = np.zeros(len(data), bool)
    in_dev[idx_dev] = True
    known = data["fu_years"].notna().to_numpy()
    base = baseline_features(cohort.data.columns, "mortality")
    cfgs = feature_configs(base)

    D = data[in_dev].reset_index(drop=True)
    yD = D[MORTALITY_TARGET].astype(int).to_numpy()
    DS = data[in_dev & known].reset_index(drop=True)
    yS = Surv.from_arrays(DS[MORTALITY_TARGET].astype(bool), DS["fu_years"].clip(lower=1 / 365.25))

    # ------------------------------------------------------------ A. feature selection
    print(f"A. Feature selection: binary dev n={len(D)} ({yD.sum()} deaths); "
          f"survival dev n={len(DS)} ({int(yS['event'].sum())} deaths); {len(base)} candidate features")
    rows_b, rows_s, oof_b, oof_s = [], [], {}, {}
    for name, cfg in cfgs.items():
        rb = repeated_cv(binary_factory(cfg), D[cfg["cols"]], yD, n_repeats=args.repeats, sensitivity_target=None,
                         name=name)
        sb = rb.summary()
        rows_b.append({"config": name, "n_inputs": n_selected(cfg), **{k: v for k, v in sb.items() if k != "model"}})
        oof_b[name] = rb.oof.mean(0)
        rs = survival_cv(name, rsf_fp(cfg), DS[cfg["cols"]], yS, args.repeats, verbose=False)
        rows_s.append({"config": name, "n_inputs": n_selected(cfg),
                       **{k: v for k, v in rs["summary"].items() if k != "model"}})
        oof_s[name] = rs["oof"]
        print(f"  {name:36s} inputs={str(cfg['n'] or '?'):>3s} | XGB AUC={sb['roc_auc_mean']:.3f}±{sb['roc_auc_sd']:.3f}"
              f" PR={sb['pr_auc_mean']:.3f} | RSF C={rs['summary']['harrell_c_mean']:.3f}"
              f"±{rs['summary']['harrell_c_sd']:.3f} AUC5y={rs['summary'].get('auc_5y_mean', np.nan):.3f}")

    fb, fs = pd.DataFrame(rows_b), pd.DataFrame(rows_s)
    for df, oof, y, metric, kind in ((fb, oof_b, yD, "roc_auc", "binary"), (fs, oof_s, yS, "harrell_c", "rsf")):
        ref = oof["all_features"]
        for i, name in enumerate(df["config"]):
            if name == "all_features":
                continue
            if kind == "binary":
                dl = delong_test(y, oof[name], ref)
                df.loc[i, "delta_vs_all"] = dl["diff"]
                df.loc[i, "p_vs_all"] = dl["p_value"]
            else:
                pc = paired_c_bootstrap(y, oof[name], ref, n_boot=500)
                df.loc[i, "delta_vs_all"] = pc["delta_harrell_c"]
                df.loc[i, "p_vs_all"] = pc["p_two_sided"]
    fb.to_csv(OUT / "feature_selection_binary.csv", index=False)
    fs.to_csv(OUT / "feature_selection_rsf.csv", index=False)
    chosen_b = one_sd_rule(fb, "roc_auc")
    chosen_s = one_sd_rule(fs, "harrell_c")
    best_b = fb.loc[fb["roc_auc_mean"].idxmax(), "config"]
    best_s = fs.loc[fs["harrell_c_mean"].idxmax(), "config"]
    print(f"  -> binary: best={best_b}, 1-SD choice={chosen_b}; RSF: best={best_s}, 1-SD choice={chosen_s}")
    stability_report(D, yD, base).to_csv(OUT / "stability_frequencies.csv", index=False)

    # ------------------------------------------------------------ B. patient selection
    print("\nB. Patient selection (each population evaluated within itself)")
    miss_frac = data[base].isna().mean(axis=1).to_numpy()
    centre_size = cohort.centre.map(cohort.centre.value_counts()).to_numpy()
    populations = {
        "all_patients": np.ones(len(data), bool),
        "without_incomplete_records": ~cohort.incomplete_record.to_numpy(),
        "missing_le_20pct": miss_frac <= 0.20,
        "centres_ge_20_patients": centre_size >= 20,
    }
    rows_p = []
    for pname, mask in populations.items():
        for fname in dict.fromkeys(["all_features", chosen_b, chosen_s]):
            cfg = cfgs[fname]
            Pb = data[in_dev & mask].reset_index(drop=True)
            yb = Pb[MORTALITY_TARGET].astype(int).to_numpy()
            rb = repeated_cv(binary_factory(cfg), Pb[cfg["cols"]], yb, n_repeats=args.repeats,
                             sensitivity_target=None).summary()
            Ps = data[in_dev & mask & known].reset_index(drop=True)
            ys = Surv.from_arrays(Ps[MORTALITY_TARGET].astype(bool), Ps["fu_years"].clip(lower=1 / 365.25))
            rs = survival_cv(fname, rsf_fp(cfg), Ps[cfg["cols"]], ys, args.repeats, verbose=False)["summary"]
            rows_p.append({"population": pname, "features": fname, "n": len(yb), "deaths": int(yb.sum()),
                           "mortality": float(yb.mean()), "xgb_auc": rb["roc_auc_mean"], "xgb_auc_sd": rb["roc_auc_sd"],
                           "xgb_pr_auc": rb["pr_auc_mean"], "rsf_c": rs["harrell_c_mean"],
                           "rsf_c_sd": rs["harrell_c_sd"], "rsf_auc5y": rs.get("auc_5y_mean")})
            r = rows_p[-1]
            print(f"  {pname:28s} {fname:24s} n={r['n']:3d} deaths={r['deaths']:3d} ({r['mortality']:.0%}) | "
                  f"XGB AUC={r['xgb_auc']:.3f} PR={r['xgb_pr_auc']:.3f} | RSF C={r['rsf_c']:.3f}")
    pd.DataFrame(rows_p).to_csv(OUT / "patient_selection.csv", index=False)

    # ------------------------------------------------------------ C. hold-out (once)
    print("\nC. Hold-out, scored once (all hold-out patients)")
    H = data[~in_dev].reset_index(drop=True)
    yH = H[MORTALITY_TARGET].astype(int).to_numpy()
    HS = data[~in_dev & known].reset_index(drop=True)
    yHS = Surv.from_arrays(HS[MORTALITY_TARGET].astype(bool), HS["fu_years"].clip(lower=1 / 365.25))
    hold: Dict[str, Any] = {"n": int(len(yH)), "deaths": int(yH.sum()), "survival_n": int(len(HS))}
    p_b, r_s = {}, {}
    for fname in dict.fromkeys(["all_features", chosen_b, chosen_s, best_b, best_s]):
        cfg = cfgs[fname]
        p_b[fname] = binary_factory(cfg)().fit(D[cfg["cols"]], yD).predict_proba(H[cfg["cols"]])[:, 1]
        r_s[fname] = survival_factory("rsf", cfg["cols"], selector=cfg["selector"])().fit(
            DS[cfg["cols"]], yS).predict(HS[cfg["cols"]])
        lo, hi = bootstrap_ci(yH, p_b[fname])
        yS_e, yH_e = truncate(yS), truncate(yHS)
        times = EVAL_TIMES[EVAL_TIMES < yH_e["time"].max()]
        auc_t, _ = cumulative_dynamic_auc(yS_e, yH_e, r_s[fname], times)
        hold[fname] = {
            "xgb_auc": float(delong_test(yH, p_b[fname], p_b[fname])["auc_1"]), "xgb_auc_ci": [lo, hi],
            "rsf_harrell_c": _harrell(yHS, r_s[fname]),
            "rsf_uno_c": float(concordance_index_ipcw(yS_e, yH_e, r_s[fname], tau=TAU)[0]),
            **{f"rsf_auc_{t:g}y": float(a) for t, a in zip(times, auc_t)},
        }
        if fname != "all_features":
            hold[fname]["xgb_delong_vs_all"] = delong_test(yH, p_b[fname], p_b["all_features"])["p_value"]
            hold[fname]["rsf_paired_vs_all"] = paired_c_bootstrap(yHS, r_s[fname], r_s["all_features"], n_boot=500)
        h = hold[fname]
        print(f"  {fname:24s} XGB AUC={h['xgb_auc']:.3f} [{lo:.3f}-{hi:.3f}] | RSF C={h['rsf_harrell_c']:.3f} "
              f"Uno C={h['rsf_uno_c']:.3f} AUC5y={h.get('rsf_auc_5y', np.nan):.3f}")

    summary = {"chosen_binary": chosen_b, "best_binary": best_b, "chosen_rsf": chosen_s, "best_rsf": best_s,
               "chosen_binary_features": cfgs[chosen_b]["cols"] if cfgs[chosen_b]["selector"] is None else "in-fold",
               "holdout": hold, "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    print(f"\nSaved to {OUT.relative_to(PROJECT_ROOT)} ({summary['seconds']} s)")
    return 0


def _harrell(y, risk) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


if __name__ == "__main__":
    raise SystemExit(main())
