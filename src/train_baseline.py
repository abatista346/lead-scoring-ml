"""Baselines for the repeat-purchase propensity model:
majority-class (illustrates why accuracy is useless here) and a
class-weighted logistic regression (the real baseline XGBoost/CatBoost
need to beat).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

NON_FEATURE_COLS = ["customer_unique_id", "target_repeat_purchase", "split"]


def load_split():
    df = pd.read_csv(PROCESSED_DIR / "model_features.csv")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    return train[feature_cols], train["target_repeat_purchase"], test[feature_cols], test["target_repeat_purchase"], feature_cols


def evaluate(name: str, y_true: pd.Series, y_pred_proba: np.ndarray) -> dict:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_proba)), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, y_pred_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "positive_rate_in_predictions": round(float(y_pred.mean()), 4),
    }
    return metrics


def main() -> None:
    X_train, y_train, X_test, y_test, _ = load_split()

    # Majority-class baseline: always predict "no repeat purchase"
    majority_proba = np.zeros(len(y_test))
    majority_metrics = evaluate("majority_class", y_test, majority_proba)
    print(majority_metrics)

    # Logistic regression with class-weighting for the imbalance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_metrics = evaluate("logistic_regression", y_test, logreg_proba)
    print(logreg_metrics)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["baseline"] = {"majority_class": majority_metrics, "logistic_regression": logreg_metrics}
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Wrote baseline metrics to {metrics_path}")


if __name__ == "__main__":
    main()
