from pathlib import Path

import pandas as pd

from fraud_detector import (
    FEATURE_NAMES,
    evaluate_model,
    generate_dataset,
    train_model,
    write_excel_report,
)


def test_generated_dataset_has_expected_imbalance_and_features():
    x, y = generate_dataset(n_samples=500, random_state=7)
    assert x.shape == (500, len(FEATURE_NAMES))
    assert x.columns.tolist() == FEATURE_NAMES
    assert y.mean() < 0.15


def test_model_evaluation_returns_fraud_metrics():
    x, y = generate_dataset(n_samples=500, random_state=7)
    train_x, test_x = x.iloc[:400], x.iloc[400:]
    train_y, test_y = y.iloc[:400], y.iloc[400:]
    model = train_model(train_x, train_y)
    metrics, predictions = evaluate_model(model, test_x, test_y)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision_pr_auc"] <= 1
    assert len(predictions) == 100
    assert "fraud_probability" in predictions


def test_excel_report_contains_expected_sheets(tmp_path: Path):
    output = tmp_path / "report.xlsx"
    metrics = {
        "roc_auc": 0.8, "average_precision_pr_auc": 0.4, "precision": 0.5,
        "recall": 0.6, "f1": 0.55, "threshold": 0.5, "test_rows": 10,
        "fraud_rate_test": 0.1, "true_negatives": 8, "false_positives": 1,
        "false_negatives": 1, "true_positives": 0,
    }
    ranking = pd.DataFrame({"rank": [1], "feature": ["device_risk_score"], "mean_abs_shap": [0.4], "mean_shap_direction": [0.2]})
    predictions = pd.DataFrame({"fraud_probability": [0.8]})
    write_excel_report(metrics, ranking, predictions, output)
    sheets = pd.ExcelFile(output).sheet_names
    assert sheets == ["Metrics", "SHAP_Ranking", "Confusion_Matrix", "Predictions"]
