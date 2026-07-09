"""Train an XGBoost classifier for repeat-purchase propensity, using
scale_pos_weight to handle the ~3% positive rate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
NON_FEATURE_COLS = ["customer_unique_id", "target_repeat_purchase", "split"]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="aucpr",
)


def load_split():
    df = pd.read_csv(PROCESSED_DIR / "model_features.csv")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    return train[feature_cols], train["target_repeat_purchase"], test[feature_cols], test["target_repeat_purchase"]


def evaluate(name: str, y_true: pd.Series, y_pred_proba: np.ndarray) -> dict:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_pred_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "positive_rate_in_predictions": round(float(y_pred.mean()), 4),
    }


def main() -> None:
    X_train, y_train, X_test, y_test = load_split()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate("xgboost", y_test, proba)
    print(metrics)

    preds_path = PROCESSED_DIR / "xgboost_test_predictions.csv"
    pd.DataFrame({"customer_unique_id": pd.read_csv(PROCESSED_DIR / "model_features.csv")
                  .loc[lambda d: d["split"] == "test", "customer_unique_id"].values,
                  "y_true": y_test.values, "y_proba": proba}).to_csv(preds_path, index=False)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["xgboost"] = metrics
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Wrote metrics to {metrics_path} and test predictions to {preds_path}")


if __name__ == "__main__":
    main()
