#!/usr/bin/env python3
"""
Step 4-5 of the model-improvement protocol: pre-registered experiments.

Development data: mortality -> 80% stratified split of the cleaned cohort
(the 20% hold-out is scored ONCE at the end); dialysis -> all patients with a
known Dializa value (n is too small for a hold-out; repeated CV only).

Each experiment: hypothesis -> repeated CV (same seeds/folds for all variants)
-> comparison with the reference (delta AUC vs SD between repeats, DeLong on
repeat-averaged OOF predictions) -> decision.

Selection rule (fixed before running): highest mean OOF ROC AUC, then the
1-SD rule - the simplest configuration within one SD of the best is chosen.

Usage:
    venv/bin/python scripts/run_experiments.py --task mortality
    venv/bin/python scripts/run_experiments.py --task dialysis
    (--quick for a fast smoke run)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import DIALYSIS_TARGET, MORTALITY_TARGET, load_cohort  # noqa: E402
from src.data.feature_sets import add_clinical_features, baseline_features  # noqa: E402
from src.models.evaluation_protocol import (  # noqa: E402
    CVResult, binary_metrics, bootstrap_ci, delong_test, grouped_cv, positive_proba, repeated_cv,
    threshold_for_sensitivity,
)
from src.models.model_zoo import MODEL_KINDS, make_factory, make_stacking_factory  # noqa: E402
from src.models.tuning import SEARCH_SPACES, nested_cv, tune  # noqa: E402

warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.getLogger("interpret").setLevel(logging.WARNING)

REPORT_ROOT = PROJECT_ROOT / "reports" / "model_improvement"
SIMPLICITY = {"logreg": 0, "ebm": 1, "random_forest": 2, "hist_gb": 3, "lightgbm": 4, "xgboost": 5,
              "catboost": 6, "stacking": 7}
SENSITIVITY_TARGET = 0.85


# --------------------------------------------------------------------------- data
def load_task(task: str) -> Dict[str, Any]:
    cohort = load_cohort()
    data = add_clinical_features(cohort.data)
    target = MORTALITY_TARGET if task == "mortality" else DIALYSIS_TARGET
    keep = data[target].notna().to_numpy()
    data = data[keep].reset_index(drop=True)
    centre = cohort.centre[keep].reset_index(drop=True)
    complete = (~cohort.incomplete_record[keep]).to_numpy()
    derived = ["eGFR_approx", "Liczba_Manifestacji_Tak", "ANCA_PR3_lub_MPO"] + [c for c in data if c.startswith("log_")]
    base = [c for c in baseline_features(data.columns, task) if c not in derived]
    return {"data": data, "y": data[target].astype(int).to_numpy(), "centre": centre, "complete": complete,
            "base_cols": base, "fe_cols": base + [c for c in derived if c in data]}


# --------------------------------------------------------------------- experiments
class Runner:
    def __init__(self, X: pd.DataFrame, y: np.ndarray, n_repeats: int, out: Path):
        self.X, self.y, self.n_repeats, self.out = X, y, n_repeats, out
        self.results: Dict[str, CVResult] = {}
        self.rows: List[Dict[str, Any]] = []

    def run(self, key: str, stage: str, hypothesis: str, factory: Callable, cols: List[str],
            config: Dict[str, Any], *, n_repeats: int | None = None) -> CVResult:
        t0 = time.time()
        res = repeated_cv(factory, self.X[cols], self.y, name=key, n_repeats=n_repeats or self.n_repeats,
                          sensitivity_target=None)
        self._log(key, stage, hypothesis, res, config, time.time() - t0)
        return res

    def add(self, key: str, stage: str, hypothesis: str, res: CVResult, config: Dict[str, Any], seconds: float):
        self._log(key, stage, hypothesis, res, config, seconds)

    def _log(self, key, stage, hypothesis, res, config, seconds):
        self.results[key] = res
        s = res.summary()
        row = {"experiment": key, "stage": stage, "hypothesis": hypothesis, "config": json.dumps(config),
               "n_repeats": len(res.per_repeat), "seconds": round(seconds, 1),
               **{k: v for k, v in s.items() if k != "model"}}
        self.rows.append(row)
        print(f"  [{stage}] {key:42s} AUC={s['roc_auc_mean']:.3f}±{s['roc_auc_sd']:.3f} "
              f"PR={s['pr_auc_mean']:.3f} Brier={s['brier_mean']:.3f} slope={s['cal_slope_mean']:.2f}"
              f" ({seconds:.0f}s)")

    def compare(self, key: str, ref: str) -> Dict[str, float]:
        a, b = self.results[key], self.results[ref]
        r = min(len(a.per_repeat), len(b.per_repeat))
        dl = delong_test(self.y, a.oof[:r].mean(0), b.oof[:r].mean(0))
        d_auc = a.per_repeat["roc_auc"][:r].mean() - b.per_repeat["roc_auc"][:r].mean()
        sd = max(a.per_repeat["roc_auc"][:r].std(ddof=1), b.per_repeat["roc_auc"][:r].std(ddof=1))
        d_brier = a.per_repeat["brier"][:r].mean() - b.per_repeat["brier"][:r].mean()
        return {"reference": ref, "delta_auc": d_auc, "sd_auc": sd, "delta_brier": d_brier,
                "delong_p": dl["p_value"]}

    def decide(self, key: str, ref: str, *, calibration_only: bool = False) -> bool:
        c = self.compare(key, ref)
        if calibration_only:
            accepted = c["delta_brier"] < 0 and c["delta_auc"] > -c["sd_auc"]
        else:
            accepted = c["delta_auc"] > c["sd_auc"] and c["delong_p"] < 0.05
        row = next(r for r in self.rows if r["experiment"] == key)
        row.update(c, decision="accept" if accepted else "reject")
        return accepted

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.out / "experiments.csv", index=False)
        np.savez_compressed(self.out / "oof_experiments.npz", y=self.y,
                            **{k: v.oof for k, v in self.results.items()})


def top_kinds(runner: Runner, keys: Dict[str, str], k: int) -> List[str]:
    scores = {kind: runner.results[key].per_repeat["roc_auc"].mean() for kind, key in keys.items()}
    return sorted(scores, key=scores.get, reverse=True)[:k]


def one_sd_choice(runner: Runner, candidates: Dict[str, Dict[str, Any]]) -> str:
    """Best mean AUC, then the simplest candidate within one SD of it."""
    means = {k: runner.results[k].per_repeat["roc_auc"].mean() for k in candidates}
    best = max(means, key=means.get)
    sd = runner.results[best].per_repeat["roc_auc"].std(ddof=1)
    within = [k for k in candidates if means[k] >= means[best] - sd]
    return min(within, key=lambda k: (SIMPLICITY[candidates[k]["kind"]], -means[k]))


# --------------------------------------------------------------------------- plots
def plot_curves(y: np.ndarray, curves: Dict[str, np.ndarray], path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        m = binary_metrics(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {m['roc_auc']:.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP {m['pr_auc']:.3f})")
        axes[2].plot(mean_pred, frac, marker="o", label=f"{name} (Brier {m['brier']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    for ax, (xl, yl, t) in zip(axes, [("1 - specificity", "sensitivity", "ROC"), ("recall", "precision", "PR"),
                                      ("predicted risk", "observed rate", "Calibration")]):
        ax.set_xlabel(xl), ax.set_ylabel(yl), ax.set_title(f"{title}: {t}"), ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mortality", "dialysis"], default="mortality")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--quick", action="store_true", help="smoke run: 2 repeats, 5 trials")
    args = parser.parse_args()
    if args.quick:
        args.repeats, args.trials = 2, 5

    out = REPORT_ROOT / args.task
    out.mkdir(parents=True, exist_ok=True)
    task = load_task(args.task)
    data, y = task["data"], task["y"]
    idx = np.arange(len(y))
    if args.task == "mortality":
        dev_idx, hold_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    else:
        dev_idx, hold_idx = idx, np.array([], dtype=int)
    X_dev, y_dev = data.iloc[dev_idx].reset_index(drop=True), y[dev_idx]
    base, fe = task["base_cols"], task["fe_cols"]
    print(f"Task={args.task}: n={len(y)} (dev {len(dev_idx)}, hold-out {len(hold_idx)}), "
          f"positives={y.sum()}, base features={len(base)}")

    R = Runner(X_dev, y_dev, args.repeats, out)
    configs: Dict[str, Dict[str, Any]] = {}

    def run_cfg(key, stage, hyp, kind, cols, n_repeats=None, **kw):
        cfg = {"kind": kind, "cols": "fe" if cols is fe else "base", **kw}
        configs[key] = cfg
        return R.run(key, stage, hyp, make_factory(kind, cols, **kw), cols, cfg, n_repeats=n_repeats)

    # S1 model families
    family = {}
    for kind in MODEL_KINDS:
        key = f"S1_{kind}"
        run_cfg(key, "S1", "model family comparison at default regularised settings", kind, base)
        family[kind] = key
    ref = "S1_logreg"
    for key in family.values():
        if key != ref:
            R.decide(key, ref)
    tops = [k for k in top_kinds(R, {k: v for k, v in family.items() if k in SEARCH_SPACES}, 3)]
    print(f"  top tunable families: {tops}")

    best_opts: Dict[str, Dict[str, Any]] = {k: {} for k in tops}
    # S2 imbalance, S3 imputation, S4 clinical features, S5 L1 selection
    for kind in tops:
        cur = family[kind]
        for stage, key_sfx, hyp, kw, calib in [
            ("S2", "smote", "SMOTE (in-fold) beats class weights", {"balance": "smote"}, False),
            ("S2", "no_weights", "unweighted training gives better calibration", {"balance": "none"}, True),
            ("S3", "iterative_imputer", "multivariate imputation beats median", {"imputer": "iterative"}, False),
            ("S4", "clinical_features", "eGFR / log-labs / organ count add signal", {"_fe": True}, False),
            ("S5", "l1_selection", "in-fold L1 feature selection reduces noise", {"select_l1_C": 0.1}, False),
        ]:
            kw = dict(kw)
            cols = fe if kw.pop("_fe", False) else base
            key = f"{stage}_{kind}_{key_sfx}"
            run_cfg(key, stage, hyp, kind, cols, **kw)
            if R.decide(key, cur, calibration_only=calib):
                best_opts[kind].update(kw if cols is base else {**kw, "_fe": True})

    # S6 nested tuning (paired with defaults on the same outer folds)
    tuned_keys = {}
    nested_repeats = min(2, args.repeats)
    for kind in tops:
        opts = dict(best_opts[kind])
        cols = fe if opts.pop("_fe", False) else base
        ref_key = f"S6_{kind}_default_{nested_repeats}x"
        run_cfg(ref_key, "S6", "reference for nested tuning (same outer folds)", kind, cols,
                n_repeats=nested_repeats, **opts)
        key = f"S6_{kind}_nested_optuna"
        t0 = time.time()
        res = nested_cv(kind, cols, X_dev[cols], y_dev, n_repeats=nested_repeats, n_trials=args.trials,
                        factory_kwargs=opts, name=key)
        cfg = {"kind": kind, "cols": "fe" if cols is fe else "base", **opts, "tuned": True}
        R.add(key, "S6", "Optuna tuning (nested CV) beats defaults", res, cfg, time.time() - t0)
        configs[key] = cfg
        if R.decide(key, ref_key):
            tuned_keys[kind] = key

    # Final params for tuned kinds: tune once on the whole development set.
    final_params: Dict[str, Dict[str, Any]] = {}
    for kind in tops:
        opts = {k: v for k, v in best_opts[kind].items() if k != "_fe"}
        cols = fe if best_opts[kind].get("_fe") else base
        final_params[kind] = (tune(kind, cols, X_dev[cols], y_dev, n_trials=args.trials, factory_kwargs=opts)
                              if kind in tuned_keys else {})

    # Candidates = each top family with its accepted options (+ tuned params if accepted)
    candidates: Dict[str, Dict[str, Any]] = {}
    for kind in tops:
        opts = {k: v for k, v in best_opts[kind].items() if k != "_fe"}
        cols = fe if best_opts[kind].get("_fe") else base
        key = f"S7_{kind}_candidate"
        run_cfg(key, "S7", "final candidate with accepted options", kind, cols, params=final_params[kind] or None,
                **opts)
        configs[key].update({"params": final_params[kind]})
        candidates[key] = configs[key]
    for kind in ("logreg", "ebm"):
        if f"S1_{kind}" not in candidates and kind not in tops:
            candidates[f"S1_{kind}"] = configs[f"S1_{kind}"]

    # S7 stacking of the top candidates
    stack_base = {}
    for key in [k for k in candidates if k.startswith("S7_")]:
        cfg = candidates[key]
        cols = fe if cfg["cols"] == "fe" else base
        kw = {k: v for k, v in cfg.items() if k in ("balance", "imputer", "select_l1_C")}
        stack_base[cfg["kind"]] = make_factory(cfg["kind"], cols, params=cfg.get("params") or None, **kw)
    if len(stack_base) >= 2:
        stack_cols = fe if any(candidates[k]["cols"] == "fe" for k in candidates if k.startswith("S7_")) else base
        R.run("S7_stacking", "S7", "OOF stacking of top candidates beats the best single model",
              make_stacking_factory(stack_base), stack_cols, {"kind": "stacking", "members": list(stack_base)},
              n_repeats=min(3, args.repeats))
        configs["S7_stacking"] = {"kind": "stacking", "members": list(stack_base),
                                  "cols": "fe" if stack_cols is fe else "base"}
        best_single = max((k for k in candidates if k.startswith("S7_")),
                          key=lambda k: R.results[k].per_repeat["roc_auc"].mean())
        if R.decide("S7_stacking", best_single):
            candidates["S7_stacking"] = configs["S7_stacking"]

    chosen = one_sd_choice(R, candidates)
    cfg = configs[chosen]
    print(f"\nSelected by 1-SD rule: {chosen} -> {cfg}")

    def chosen_factory(calibration=None):
        cols = fe if cfg["cols"] == "fe" else base
        if cfg["kind"] == "stacking":
            return make_stacking_factory(stack_base), cols
        kw = {k: v for k, v in cfg.items() if k in ("balance", "imputer", "select_l1_C")}
        return make_factory(cfg["kind"], cols, params=cfg.get("params") or None, calibration=calibration, **kw), cols

    # S8 calibration of the chosen configuration
    final_calibration = None
    if cfg["kind"] != "stacking":
        for method in ("sigmoid", "isotonic"):
            fac, cols = chosen_factory(method)
            key = f"S8_{chosen}_{method}"
            R.run(key, "S8", f"{method} calibration improves Brier without AUC loss", fac, cols,
                  {**cfg, "calibration": method})
            if R.decide(key, chosen, calibration_only=True) and final_calibration is None:
                final_calibration = method
    R.save()

    # Final evaluation with fold-internal thresholds on the development data
    fac, cols = chosen_factory(final_calibration)
    final_cv = repeated_cv(fac, X_dev[cols], y_dev, name="final", n_repeats=args.repeats,
                           sensitivity_target=SENSITIVITY_TARGET)
    summary: Dict[str, Any] = {
        "task": args.task, "chosen_experiment": chosen, "config": cfg, "calibration": final_calibration,
        "n_features_used": len(cols), "features": cols,
        "dev_cv": final_cv.summary(),
    }

    curves = {"final (dev OOF)": final_cv.oof.mean(0)}
    if "S1_logreg" in R.results and chosen != "S1_logreg":
        curves["logreg baseline (dev OOF)"] = R.results["S1_logreg"].oof.mean(0)

    # Hold-out: touched once. Threshold from training-fold OOF predictions.
    if len(hold_idx):
        X_hold, y_hold = data.iloc[hold_idx].reset_index(drop=True), y[hold_idx]
        model = fac().fit(X_dev[cols], y_dev)
        p_inner = cross_val_predict(fac(), X_dev[cols], y_dev,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
        thr = threshold_for_sensitivity(y_dev, p_inner, SENSITIVITY_TARGET)
        p_hold = positive_proba(model, X_hold[cols])
        lo, hi = bootstrap_ci(y_hold, p_hold)
        summary["holdout"] = {**binary_metrics(y_hold, p_hold, thr), "threshold": thr,
                              "roc_auc_ci_low": lo, "roc_auc_ci_high": hi, "n": int(len(y_hold)),
                              "positives": int(y_hold.sum())}
        plot_curves(y_hold, {"final (hold-out)": p_hold}, out / "holdout_curves.png", f"{args.task} hold-out")

    # Sensitivity analyses on the development data
    loco = grouped_cv(fac, X_dev[cols], y_dev, task["centre"].iloc[dev_idx].reset_index(drop=True), name="final")
    loco["per_centre"].to_csv(out / "final_loco_per_centre.csv", index=False)
    summary["leave_one_centre_out"] = {"pooled_roc_auc": loco["pooled_roc_auc"], **loco["pooled"]}
    complete_dev = task["complete"][dev_idx]
    if (~complete_dev).sum() > 0:
        cv_c = repeated_cv(fac, X_dev[cols][complete_dev].reset_index(drop=True), y_dev[complete_dev],
                           name="final_complete", n_repeats=args.repeats, sensitivity_target=None)
        summary["without_incomplete_records"] = cv_c.summary()

    plot_curves(y_dev, curves, out / "dev_oof_curves.png", f"{args.task} development OOF")
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k not in ("features",)}, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
