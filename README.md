# Day 11 — Machine Learning Fraud Detection & SHAP Explainer

An end-to-end educational fraud-detection pipeline that generates an imbalanced transaction dataset, trains an XGBoost classifier, evaluates ROC-AUC and precision-recall metrics, produces SHAP feature explanations, and writes a multi-sheet Excel report.

Hugging Face is included as an **optional model/report registry**. The XGBoost model runs locally and does not require Hugging Face inference credits. With `--upload-to-hf`, the trained XGBoost artifact and reports can be uploaded to a Hugging Face model repository using `HF_TOKEN`.

> The data is synthetic for learning. This is not a production fraud model and must not be used to make real financial, account, or law-enforcement decisions.

## What it demonstrates

- Imbalanced binary classification with scikit-learn and XGBoost.
- ROC-AUC and average precision / PR-AUC.
- Precision, recall, F1, confusion matrix, and threshold handling.
- SHAP TreeExplainer global feature importance.
- PNG charts and text-based SHAP rankings.
- Excel report with Metrics, SHAP_Ranking, Confusion_Matrix, and Predictions sheets.
- Optional Hugging Face Hub artifact upload without hosted inference.

## Architecture

See [docs/architecture.md](docs/architecture.md).

```text
Synthetic transactions
        ↓
Train/test split with stratification
        ↓
XGBoost classifier with imbalance weighting
        ↓
Probability metrics + threshold predictions
        ↓
SHAP TreeExplainer
        ↓
Excel report + JSON metrics + PNG charts + model JSON
        ↓
Optional Hugging Face Hub upload
```

## Windows PowerShell setup

```powershell
git clone https://github.com/shallow2024/Day11-ml-fraud-detection-shap-explainer.git
cd D:\Day11-ml-fraud-detection-shap-explainer

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run locally

```powershell
.\.venv\Scripts\python.exe fraud_detector.py
```

The default run uses 4,000 synthetic transactions and writes to `outputs\`:

```text
outputs\fraud_detection_report.xlsx
outputs\metrics.json
outputs\fraud_xgboost.json
outputs\shap_feature_importance.png
outputs\shap_summary.png
```

The preferred Excel file contains:

- `Metrics`: ROC-AUC, PR-AUC, precision, recall, F1, threshold, and confusion counts.
- `SHAP_Ranking`: global mean absolute SHAP values and average direction.
- `Confusion_Matrix`: true/false positive and negative counts.
- `Predictions`: test rows, probabilities, and predictions.

Change the sample size or decision threshold:

```powershell
.\.venv\Scripts\python.exe fraud_detector.py `
  --samples 8000 `
  --threshold 0.35
```

## Upload to Hugging Face Hub

The model is uploaded as an XGBoost JSON artifact, not as a Transformers model. This does not call hosted inference and therefore does not use Inference Provider credits.

Create a Hugging Face model repository name such as `your-username/day11-fraud-xgboost`, then set your token in the current PowerShell session:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Run:

```powershell
.\.venv\Scripts\python.exe fraud_detector.py `
  --upload-to-hf "your-username/day11-fraud-xgboost"
```

The upload includes the model JSON, Excel report, metrics JSON, and SHAP PNG charts. Keep the token private and do not commit it.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests are API-free and validate dataset shape/imbalance, metrics, and Excel sheet structure.

## Understanding SHAP

SHAP values estimate how each feature moves a model prediction relative to a baseline. The absolute value measures contribution magnitude; the signed mean gives a rough direction across the explained sample. A high SHAP value indicates model influence, not causation or proof that a transaction is fraudulent.

## Limitations

- Synthetic data does not represent real fraud behavior.
- Imbalanced metrics must be interpreted together; ROC-AUC alone can look strong even when positive-class performance is weak.
- The default threshold of 0.5 is not automatically optimal for fraud operations.
- SHAP explains the trained model, not the real-world truth.
- Production use requires leakage controls, temporal validation, calibration, monitoring, privacy review, human governance, and an appropriate cost matrix.

## License

MIT. See [LICENSE](LICENSE).
