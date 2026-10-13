"""Day 11: synthetic fraud detection with XGBoost and SHAP explanations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_NAMES = [
    "transaction_amount",
    "hour_of_day",
    "merchant_distance_km",
    "transactions_last_hour",
    "card_age_days",
    "device_risk_score",
    "international_transaction",
    "cash_withdrawal",
]


def generate_dataset(n_samples: int = 4000, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a reproducible, intentionally imbalanced educational dataset."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=6,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.97, 0.03],
        class_sep=1.25,
        flip_y=0.01,
        random_state=random_state,
    )
    features = pd.DataFrame(x, columns=FEATURE_NAMES)
    # Put the synthetic feature scale closer to transaction concepts for readability.
    features["transaction_amount"] = np.abs(features["transaction_amount"] * 75 + 80)
    features["hour_of_day"] = np.mod(np.round((features["hour_of_day"] + 3) * 4), 24)
    features["merchant_distance_km"] = np.abs(features["merchant_distance_km"] * 25 + 20)
    features["transactions_last_hour"] = np.maximum(0, np.round(features["transactions_last_hour"] * 2 + 2))
    features["card_age_days"] = np.maximum(1, np.round((features["card_age_days"] + 2) * 300))
    features["device_risk_score"] = 1 / (1 + np.exp(-features["device_risk_score"]))
    features["international_transaction"] = (features["international_transaction"] > 0).astype(int)
    features["cash_withdrawal"] = (features["cash_withdrawal"] > 0).astype(int)
    return features, pd.Series(y, name="is_fraud")


def train_model(x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> XGBClassifier:
    negative, positive = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(negative / max(positive, 1)),
        random_state=random_state,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "fraud_rate_test": float(y_test.mean()),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    predictions_frame = x_test.copy()
    predictions_frame["actual_is_fraud"] = y_test.to_numpy()
    predictions_frame["fraud_probability"] = probabilities
    predictions_frame["predicted_is_fraud"] = predictions
    return metrics, predictions_frame


def shap_explanations(model: XGBClassifier, x_test: pd.DataFrame, output_dir: Path, max_rows: int = 500) -> tuple[pd.DataFrame, Path, Path]:
    """Generate global SHAP rankings and a summary/bar chart."""
    import shap

    sample = x_test.head(max_rows)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    ranking = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap_direction": values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking[["rank", "feature", "mean_abs_shap", "mean_shap_direction"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    bar_path = output_dir / "shap_feature_importance.png"
    plt.figure(figsize=(9, 5))
    plt.barh(ranking["feature"][::-1], ranking["mean_abs_shap"][::-1], color="#306998")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global feature importance — SHAP")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()
    summary_path = output_dir / "shap_summary.png"
    shap.summary_plot(values, sample, show=False, plot_size=(9, 5))
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()
    return ranking, bar_path, summary_path


def write_excel_report(metrics: dict[str, Any], ranking: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> None:
    """Write metrics, SHAP ranking, predictions, and confusion matrix to Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = [{"metric": key, "value": value} for key, value in metrics.items() if not isinstance(value, dict)]
    matrix = pd.DataFrame(
        [[metrics["true_negatives"], metrics["false_positives"]], [metrics["false_negatives"], metrics["true_positives"]]],
        index=["actual_normal", "actual_fraud"], columns=["predicted_normal", "predicted_fraud"],
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(metric_rows).to_excel(writer, sheet_name="Metrics", index=False)
        ranking.to_excel(writer, sheet_name="SHAP_Ranking", index=False)
        matrix.to_excel(writer, sheet_name="Confusion_Matrix")
        predictions.head(1000).to_excel(writer, sheet_name="Predictions", index=False)


def upload_to_huggingface(output_dir: Path, repo_id: str, model_path: Path) -> str:
    """Upload the XGBoost model and reports to a Hugging Face model repository."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for --upload-to-hf")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo=model_path.name, repo_id=repo_id, repo_type="model")
    for path in output_dir.glob("*"):
        if path.is_file() and path.name != model_path.name:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def run(output_dir: Path, n_samples: int, test_size: float, random_state: int, threshold: float, upload_repo: str | None = None) -> dict[str, Any]:
    x, y = generate_dataset(n_samples=n_samples, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)
    model = train_model(x_train, y_train, random_state=random_state)
    metrics, predictions = evaluate_model(model, x_test, y_test, threshold=threshold)
    ranking, _, _ = shap_explanations(model, x_test, output_dir)
    model_path = output_dir / "fraud_xgboost.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    write_excel_report(metrics, ranking, predictions, output_dir / "fraud_detection_report.xlsx")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result = {"metrics": metrics, "shap_ranking": ranking.to_dict(orient="records"), "output_dir": str(output_dir)}
    if upload_repo:
        result["huggingface_url"] = upload_to_huggingface(output_dir, upload_repo, model_path)
    print(json.dumps({"roc_auc": metrics["roc_auc"], "average_precision_pr_auc": metrics["average_precision_pr_auc"], "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "outputs": str(output_dir)}, indent=2))
    print("\nSHAP feature ranking:")
    print(ranking.to_string(index=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detector and explain it with SHAP.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--upload-to-hf", metavar="OWNER/REPO", help="Optional HF model repo for XGBoost model and reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir, args.samples, args.test_size, args.random_state, args.threshold, args.upload_to_hf)
