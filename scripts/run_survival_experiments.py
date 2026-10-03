#!/usr/bin/env python3
"""
Follow-up to reports/model_improvement/REPORT.md: better-defined outcomes.

Part 1 - survival analysis (prediction at diagnosis)
    time  = Wiek - Wiek_rozpoznania (years from diagnosis to death / last contact)
    event = Zgon
    Patients with unknown follow-up are excluded (n=51, 46 of them incomplete records).
    Models: binary XGBoost from the main report (reference, its probability used as a
    risk score), penalised Cox (Coxnet), Random Survival Forest, gradient-boosted Cox.
    Metrics on out-of-fold predictions (5x3 CV): Harrell C, Uno C (tau = 10 y),
    time-dependent AUC at 1 / 3 / 5 years (IPCW, cumulative/dynamic).
    Fixed-horizon variant: binary 5-year mortality, censored-before-5-years excluded.

Part 2 - landmark model at 6 months
    Patients alive and under follow-up 6 months after diagnosis. Features at the
    landmark = diagnosis features + treatment at 6 months (Leczenie_6m_*). Compared
    with diagnosis-only features ON THE SAME PATIENTS (paired DeLong / C-index).

The 20% hold-out (same split as scripts/run_experiments.py) is scored once for the
selected survival model.

Usage: venv/bin/python scripts/run_survival_experiments.py [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import MORTALITY_TARGET, load_cohort  # noqa: E402
from src.data.feature_sets import baseline_features  # noqa: E402
from src.models.evaluation_protocol import bootstrap_ci, delong_test, repeated_cv  # noqa: E402
from src.models.model_zoo import make_factory  # noqa: E402
from src.models.survival import (  # noqa: E402
    EVAL_TIMES, TAU, coxnet_fp, paired_c_bootstrap, sksurv_fp, survival_cv, survival_factory, truncate,
)

warnings.filterwarnings("ignore")
OUT = PROJECT_ROOT / "reports" / "model_improvement" / "survival"
LANDMARK = 0.5  # years


def binary_fp(cols):
    """Reference: the binary classifier chosen in the main report (risk = P(death))."""
    def fp(X_tr, y_tr, X_te):
        model = make_factory("xgboost", cols, balance="none")().fit(X_tr, y_tr["event"].astype(int))
        return model.predict_proba(X_te)[:, 1]
    return fp


# ------------------------------------------------------------------------------ main
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
    results: Dict[str, Dict] = {"notes": {
        "follow_up_unknown_excluded": int((~known).sum()),
        "of_which_incomplete_records": int((~known & cohort.incomplete_record.to_numpy()).sum()),
    }}

    # ---------------------------------------------------------------- Part 1: survival
    dev = in_dev & known
    D = data[dev].reset_index(drop=True)
    yS = Surv.from_arrays(D[MORTALITY_TARGET].astype(bool), D["fu_years"].clip(lower=1 / 365.25))
    X = D[base]
    print(f"Part 1 - survival: dev n={len(D)}, deaths={int(yS['event'].sum())}, "
          f"median follow-up={np.median(yS['time']):.1f} y")
    part1 = {}
    part1["binary_xgboost_reference"] = survival_cv("binary XGBoost (reference, P(death) as risk)",
                                                    binary_fp(base), X, yS, args.repeats)
    part1["coxnet"] = survival_cv("Cox elastic-net (alpha tuned in-fold)", coxnet_fp(base), X, yS, args.repeats)
    part1["rsf"] = survival_cv("Random Survival Forest", sksurv_fp(survival_factory("rsf", base)), X, yS, args.repeats)
    part1["gb_cox"] = survival_cv("Gradient-boosted Cox", sksurv_fp(survival_factory("gb_cox", base)), X, yS,
                                  args.repeats)
    for k, v in part1.items():
        if k != "binary_xgboost_reference":
            v["summary"]["paired_vs_binary"] = paired_c_bootstrap(yS, v["oof"], part1["binary_xgboost_reference"]["oof"])
            ref = part1["binary_xgboost_reference"]["summary"]
            v["summary"]["delta_uno_c_vs_binary"] = v["summary"]["uno_c_mean"] - ref["uno_c_mean"]
            v["summary"]["delta_auc_5y_vs_binary"] = v["summary"].get("auc_5y_mean", np.nan) - ref.get("auc_5y_mean", np.nan)

    # Fixed 5-year horizon binary target (censored < 5 y excluded)
    h = 5.0
    elig = ~((D[MORTALITY_TARGET] == 0) & (D["fu_years"] < h))
    Xh = D.loc[elig, base].reset_index(drop=True)
    yh = ((D.loc[elig, MORTALITY_TARGET] == 1) & (D.loc[elig, "fu_years"] <= h)).astype(int).to_numpy()
    res_h = repeated_cv(make_factory("xgboost", base, balance="none"), Xh, yh, n_repeats=args.repeats,
                        sensitivity_target=None, name="xgb_5y")
    res_h_lr = repeated_cv(make_factory("logreg", base), Xh, yh, n_repeats=args.repeats, sensitivity_target=None,
                           name="lr_5y")
    part1["fixed_horizon_5y"] = {"summary": {"n": int(len(yh)), "events": int(yh.sum()),
                                             "xgboost": res_h.summary(), "logreg": res_h_lr.summary()}}
    print(f"  {'binary 5-year mortality (XGBoost)':44s} AUC={res_h.summary()['roc_auc_mean']:.3f}"
          f"±{res_h.summary()['roc_auc_sd']:.3f} (n={len(yh)}, events={yh.sum()}); "
          f"LR AUC={res_h_lr.summary()['roc_auc_mean']:.3f}")

    # Hold-out: best survival model by Uno C, scored once
    surv_keys = ["coxnet", "rsf", "gb_cox"]
    best = max(surv_keys, key=lambda k: part1[k]["summary"]["uno_c_mean"])
    hold = ~in_dev & known
    H = data[hold].reset_index(drop=True)
    yH = Surv.from_arrays(H[MORTALITY_TARGET].astype(bool), H["fu_years"].clip(lower=1 / 365.25))
    fp = {"coxnet": coxnet_fp(base), "rsf": sksurv_fp(survival_factory("rsf", base)),
          "gb_cox": sksurv_fp(survival_factory("gb_cox", base))}[best]
    risk_h = fp(X, yS, H[base])
    risk_ref = binary_fp(base)(X, yS, H[base])
    yS_eval, yH_eval = truncate(yS), truncate(yH)
    times = EVAL_TIMES[EVAL_TIMES < yH_eval["time"].max()]
    auc_t, _ = cumulative_dynamic_auc(yS_eval, yH_eval, risk_h, times)
    auc_ref, _ = cumulative_dynamic_auc(yS_eval, yH_eval, risk_ref, times)
    results["holdout"] = {
        "model": best, "n": int(len(H)), "deaths": int(yH["event"].sum()),
        "uno_c": float(concordance_index_ipcw(yS_eval, yH_eval, risk_h, tau=TAU)[0]),
        "uno_c_binary_reference": float(concordance_index_ipcw(yS_eval, yH_eval, risk_ref, tau=TAU)[0]),
        **{f"auc_{t:g}y": float(a) for t, a in zip(times, auc_t)},
        **{f"auc_{t:g}y_binary_reference": float(a) for t, a in zip(times, auc_ref)},
    }
    print(f"  hold-out ({best}): {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in results['holdout'].items()})}")
    results["part1_survival"] = {k: v["summary"] for k, v in part1.items()}

    # Sensitivity: without incomplete records (same models, same protocol)
    comp = ~cohort.incomplete_record.to_numpy()[dev]
    Dc = D[comp].reset_index(drop=True)
    yc = yS[comp]
    sens = {
        "binary_xgboost_reference": survival_cv("[no incomplete] binary XGBoost", binary_fp(base), Dc[base], yc,
                                                args.repeats),
        best: survival_cv(f"[no incomplete] {best}", fp, Dc[base], yc, args.repeats),
    }
    sens[best]["summary"]["paired_vs_binary"] = paired_c_bootstrap(yc, sens[best]["oof"],
                                                                   sens["binary_xgboost_reference"]["oof"])
    results["part1_without_incomplete"] = {k: v["summary"] for k, v in sens.items()}

    # ---------------------------------------------------------------- Part 2: landmark
    lm = known & ~(data["fu_years"] < LANDMARK).to_numpy()
    L = data[lm & in_dev].reset_index(drop=True)
    six_m = [c for c in cohort.data.columns if c.startswith("Leczenie_6m_")]
    yL = L[MORTALITY_TARGET].astype(int).to_numpy()
    print(f"\nPart 2 - landmark 6 months: dev n={len(L)}, deaths={yL.sum()}, added features={len(six_m)}")
    part2, landmark_oof = {}, {}
    for kind in ("logreg", "xgboost"):
        kw = {"balance": "none"} if kind == "xgboost" else {}
        r_base = repeated_cv(make_factory(kind, base, **kw), L[base], yL, n_repeats=args.repeats,
                             sensitivity_target=None, name=f"{kind}_diagnosis")
        r_lm = repeated_cv(make_factory(kind, base + six_m, **kw), L[base + six_m], yL, n_repeats=args.repeats,
                           sensitivity_target=None, name=f"{kind}_landmark")
        dl = delong_test(yL, r_lm.oof.mean(0), r_base.oof.mean(0))
        landmark_oof[kind] = r_lm.oof.mean(0)
        part2[kind] = {"diagnosis_features": r_base.summary(), "landmark_features": r_lm.summary(), "delong": dl}
        print(f"  {kind:8s} diagnosis AUC={r_base.summary()['roc_auc_mean']:.3f}±{r_base.summary()['roc_auc_sd']:.3f}"
              f" | +6m treatment AUC={r_lm.summary()['roc_auc_mean']:.3f}±{r_lm.summary()['roc_auc_sd']:.3f}"
              f" | ΔAUC={dl['diff']:+.3f} p={dl['p_value']:.3f}")
    # survival from the landmark, same comparison
    yLs = Surv.from_arrays(L[MORTALITY_TARGET].astype(bool), (L["fu_years"] - LANDMARK).clip(lower=1 / 365.25))
    s_base = survival_cv("landmark survival, diagnosis features (RSF)",
                         sksurv_fp(survival_factory("rsf", base)), L[base], yLs, args.repeats)
    s_lm = survival_cv("landmark survival, +6m treatment (RSF)",
                       sksurv_fp(survival_factory("rsf", base + six_m)), L[base + six_m], yLs, args.repeats)
    part2["rsf_survival"] = {"diagnosis_features": s_base["summary"], "landmark_features": s_lm["summary"],
                             "paired": paired_c_bootstrap(yLs, s_lm["oof"], s_base["oof"])}
    compL = ~cohort.incomplete_record.to_numpy()[lm & in_dev]
    Lc, yLc = L[compL].reset_index(drop=True), yL[compL]
    rb = repeated_cv(make_factory("xgboost", base, balance="none"), Lc[base], yLc, n_repeats=args.repeats,
                     sensitivity_target=None)
    rl = repeated_cv(make_factory("xgboost", base + six_m, balance="none"), Lc[base + six_m], yLc,
                     n_repeats=args.repeats, sensitivity_target=None)
    part2["xgboost_without_incomplete"] = {
        "n": int(len(yLc)), "deaths": int(yLc.sum()),
        "diagnosis_auc": rb.summary()["roc_auc_mean"], "landmark_auc": rl.summary()["roc_auc_mean"],
        "delong": delong_test(yLc, rl.oof.mean(0), rb.oof.mean(0)),
    }
    print(f"  [no incomplete] xgboost diagnosis AUC={rb.summary()['roc_auc_mean']:.3f} | +6m "
          f"AUC={rl.summary()['roc_auc_mean']:.3f} | p={part2['xgboost_without_incomplete']['delong']['p_value']:.3f}")
    lo, hi = bootstrap_ci(yL, landmark_oof["xgboost"])
    part2["xgboost"]["landmark_auc_ci"] = [lo, hi]
    # Landmark hold-out (scored once): diagnosis vs +6m features, same patients
    Lh = data[lm & ~in_dev].reset_index(drop=True)
    yLh = Lh[MORTALITY_TARGET].astype(int).to_numpy()
    yLhs = Surv.from_arrays(Lh[MORTALITY_TARGET].astype(bool), (Lh["fu_years"] - LANDMARK).clip(lower=1 / 365.25))
    hold_lm = {"n": int(len(yLh)), "deaths": int(yLh.sum())}
    for label, cols in (("diagnosis", base), ("landmark", base + six_m)):
        m = make_factory("xgboost", cols, balance="none")().fit(L[cols], yL)
        p_h = m.predict_proba(Lh[cols])[:, 1]
        lo_h, hi_h = bootstrap_ci(yLh, p_h)
        rsf = survival_factory("rsf", cols)().fit(L[cols], yLs)
        hold_lm[label] = {"xgb_auc": float(delong_test(yLh, p_h, p_h)["auc_1"]), "xgb_auc_ci": [lo_h, hi_h],
                          "rsf_harrell_c": float(concordance_index_censored(yLhs["event"], yLhs["time"],
                                                                            rsf.predict(Lh[cols]))[0]),
                          "_p": p_h}
    hold_lm["delong_landmark_vs_diagnosis"] = delong_test(yLh, hold_lm["landmark"].pop("_p"),
                                                          hold_lm["diagnosis"].pop("_p"))
    part2["holdout"] = hold_lm
    print(f"  hold-out landmark: n={len(yLh)} deaths={yLh.sum()} XGB AUC diagnosis="
          f"{hold_lm['diagnosis']['xgb_auc']:.3f} -> +6m {hold_lm['landmark']['xgb_auc']:.3f} "
          f"(p={hold_lm['delong_landmark_vs_diagnosis']['p_value']:.3f}); RSF C "
          f"{hold_lm['diagnosis']['rsf_harrell_c']:.3f} -> {hold_lm['landmark']['rsf_harrell_c']:.3f}")
    results["part2_landmark"] = part2
    results["seconds"] = round(time.time() - t0)

    (OUT / "survival_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    print(f"\nSaved {OUT / 'survival_summary.json'} ({results['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
