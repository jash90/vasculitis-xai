#!/usr/bin/env python3
"""
Fit and save the dialysis model chosen by scripts/run_experiments.py --task dialysis.

Target: Dializa == 1 vs 2 among patients with a recorded value (renal patients).
The cohort is too small for a hold-out, so performance comes from repeated CV
(reports/model_improvement/dialysis/final_summary.json); the saved model is fitted
on all eligible patients.

Sensitivity analysis (--all-patients-sensitivity): treats unrecorded Dializa as
"no dialysis" on the full cohort. This mostly measures who had the field filled
in (renal patients), so it is reported but never saved as a model.

Outputs: models/saved_dialysis/{dialysis_model.joblib, dialysis_feature_names.json,
dialysis_metadata.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import DIALYSIS_TARGET, load_cohort  # noqa: E402
from src.data.feature_sets import add_clinical_features, baseline_features  # noqa: E402
from src.models.evaluation_protocol import repeated_cv  # noqa: E402
from src.models.model_zoo import make_factory, make_stacking_factory  # noqa: E402

warnings.filterwarnings("ignore")
SUMMARY = PROJECT_ROOT / "reports" / "model_improvement" / "dialysis" / "final_summary.json"
OUT_DIR = PROJECT_ROOT / "models" / "saved_dialysis"


def build_factory(summary: dict, cols: list):
    cfg = summary["config"]
    if cfg["kind"] == "stacking":
        members = {k: make_factory(k, cols) for k in cfg["members"]}
        return make_stacking_factory(members)
    kw = {k: v for k, v in cfg.items() if k in ("balance", "imputer", "select_l1_C")}
    return make_factory(cfg["kind"], cols, params=cfg.get("params") or None,
                        calibration=summary.get("calibration"), **kw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-patients-sensitivity", action="store_true")
    args = parser.parse_args()

    if not SUMMARY.exists():
        print(f"Missing {SUMMARY}; run scripts/run_experiments.py --task dialysis first.")
        return 1
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cols = summary["features"]

    cohort = load_cohort()
    data = add_clinical_features(cohort.data)
    known = data[DIALYSIS_TARGET].notna()
    X, y = data.loc[known, cols].reset_index(drop=True), data.loc[known, DIALYSIS_TARGET].astype(int).to_numpy()
    factory = build_factory(summary, cols)

    model = factory().fit(X, y)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "dialysis_model.joblib")
    (OUT_DIR / "dialysis_feature_names.json").write_text(json.dumps(cols, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    metadata = {
        "target": "Dializa == 1 (vs 2) among patients with a recorded value",
        "n_patients": int(len(y)), "positives": int(y.sum()),
        "config": summary["config"], "calibration": summary.get("calibration"),
        "cv_performance": summary["dev_cv"],
        "leave_one_centre_out": summary.get("leave_one_centre_out"),
        "input_coding": "registry_recoded_v2 (src.data.cohort.recode + add_clinical_features)",
    }

    if args.all_patients_sensitivity:
        y_all = (data[DIALYSIS_TARGET] == 1).astype(int).to_numpy()
        base = [c for c in baseline_features(data.columns, "dialysis") if c in data.columns]
        res = repeated_cv(make_factory("logreg", base), data[base], y_all, n_repeats=3, sensitivity_target=None)
        metadata["sensitivity_all_patients_unrecorded_as_no"] = {
            "warning": "target mostly encodes whether the field was recorded (renal patients); not a valid model",
            **res.summary(),
        }
        print(f"  all-patients sensitivity AUC = {res.per_repeat['roc_auc'].mean():.3f}")

    (OUT_DIR / "dialysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                               default=str), encoding="utf-8")
    cv = summary["dev_cv"]
    print(f"Saved dialysis model ({summary['config']['kind']}, n={len(y)}, positives={y.sum()}); "
          f"CV AUC={cv['roc_auc_mean']:.3f}±{cv['roc_auc_sd']:.3f} -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
