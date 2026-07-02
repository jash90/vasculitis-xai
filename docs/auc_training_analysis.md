# AUC Training Analysis

## Root Cause

The project has two different model paths:

- API path: 20 features collected by `PatientInput` and served by the FastAPI application.
- AUC-first offline path: 71 selected features from the richer clinical CSV.

The reference table with AUC around `0.90-0.91` is only reachable in the
offline path. The 20-feature API artifacts currently evaluate around `0.79-0.83`
AUC on their saved test split, because the form does not collect many of the
clinical variables used by the richer models.

During analysis, a direct CSV experiment accidentally allowed `Kod` into feature
selection. That raised AUC but is patient identifier leakage. The new training
path removes identifier columns before feature scoring and fits imputation,
feature selection, and scaling only on the training split.

## Current AUC-First Result

Command:

```bash
./venv/bin/python scripts/train_auc_models.py
```

Output artifacts:

- `models/saved/auc/best_auc_model.joblib`
- `models/saved/auc/auc_model_comparison.json`
- `models/saved/auc/auc_evaluation_report.json`
- `models/saved/auc/feature_names_auc.json`
- `models/saved/auc/X_train_auc.joblib`
- `models/saved/auc/X_test_auc.joblib`
- `models/saved/auc/y_train_auc.joblib`
- `models/saved/auc/y_test_auc.joblib`

Latest holdout metrics:

| Model | AUC | AP | Accuracy | Recall | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stacking | 0.912 | 0.790 | 0.856 | 0.711 | 0.894 | 0.643 | 0.675 |
| catboost | 0.908 | 0.779 | 0.883 | 0.632 | 0.951 | 0.774 | 0.696 |
| lightgbm | 0.899 | 0.764 | 0.856 | 0.605 | 0.923 | 0.676 | 0.639 |
| svm | 0.899 | 0.798 | 0.883 | 0.632 | 0.951 | 0.774 | 0.696 |
| random_forest | 0.895 | 0.774 | 0.883 | 0.579 | 0.965 | 0.815 | 0.677 |
| logistic_regression | 0.883 | 0.711 | 0.822 | 0.763 | 0.838 | 0.558 | 0.644 |
| gradient_boosting | 0.878 | 0.771 | 0.883 | 0.553 | 0.972 | 0.840 | 0.667 |
| xgboost | 0.873 | 0.749 | 0.861 | 0.605 | 0.930 | 0.697 | 0.648 |
| neural_network | 0.758 | 0.626 | 0.844 | 0.342 | 0.979 | 0.812 | 0.481 |

## Interpretation

The best current model is `stacking` with `AUC=0.912`, close to the reference
CatBoost `AUC=0.914` and above the reference stacking `AUC=0.906`. CatBoost
itself reaches `AUC=0.908` in the leakage-safe path.

Do not replace the API model with these artifacts unless the API contract is
extended to collect the 71 selected features. Otherwise the model will either
fail feature validation or silently receive zeros for unavailable fields.
