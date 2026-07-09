"""ROC/PR curve comparison across all three models, plus feature
importance for CatBoost (the winning model). Reuses the persisted
out-of-sample test predictions from train_xgboost.py and
train_catboost.py; refits logistic regression (cheap) since its
predictions weren't persisted, and refits CatBoost once more to read
off feature importances.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
NON_FEATURE_COLS = ["customer_unique_id", "target_repeat_purchase", "split"]

COLOR_ACCENT = "#1D5FD6"
COLOR_TEXT_SECONDARY = "#4A5B6E"
COLOR_MUTED = "#9AAABB"


def load_split():
    df = pd.read_csv(PROCESSED_DIR / "model_features.csv")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    return train[feature_cols], train["target_repeat_purchase"], test[feature_cols], test["target_repeat_purchase"], feature_cols


def get_logreg_proba(X_train, y_train, X_test) -> pd.Series:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_test_scaled)[:, 1]


def main() -> None:
    X_train, y_train, X_test, y_test, feature_cols = load_split()

    lr_proba = get_logreg_proba(X_train, y_train, X_test)
    xgb_preds = pd.read_csv(PROCESSED_DIR / "xgboost_test_predictions.csv")
    cat_preds = pd.read_csv(PROCESSED_DIR / "catboost_test_predictions.csv")

    models = {
        "Logistic regression": (y_test.values, lr_proba, COLOR_MUTED),
        "XGBoost": (xgb_preds["y_true"].values, xgb_preds["y_proba"].values, COLOR_TEXT_SECONDARY),
        "CatBoost": (cat_preds["y_true"].values, cat_preds["y_proba"].values, COLOR_ACCENT),
    }

    plot_roc_curves(models)
    plot_pr_curves(models, baseline_rate=y_test.mean())
    plot_catboost_feature_importance(X_train, y_train, feature_cols)


def plot_roc_curves(models: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_proba, color) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], color="#D7E3F0", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — CatBoost leads narrowly", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_roc_curves.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(models: dict, baseline_rate: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_proba, color) in models.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(recall, precision, color=color, linewidth=2, label=name)
    ax.axhline(baseline_rate, color="#D7E3F0", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves — the real test given 3% positives", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_catboost_feature_importance(X_train, y_train, feature_cols) -> None:
    model = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
        random_state=42, verbose=False,
    )
    model.fit(X_train, y_train)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values().tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance.index, importance.values, color=COLOR_ACCENT)
    ax.set_xlabel("Feature importance")
    ax.set_title("What drives the CatBoost propensity score (top 15)", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
