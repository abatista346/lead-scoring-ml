"""Segment customers into 5 propensity tiers using the CatBoost
out-of-sample predictions (the winning model from train_catboost.py).

Deliberately scores only the held-out test set rather than refitting on
the full population — every repeat-purchase rate reported per tier is
therefore genuinely out-of-sample, not inflated by the model having
seen those customers during training. In production you'd retrain on
all available data before scoring new/current leads with this same
tiering logic.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

COLOR_ACCENT = "#1D5FD6"
TIER_LABELS = ["1 - Very Low", "2 - Low", "3 - Medium", "4 - High", "5 - Very High"]


def main() -> None:
    preds = pd.read_csv(PROCESSED_DIR / "catboost_test_predictions.csv")

    preds["tier"] = pd.qcut(preds["y_proba"], 5, labels=TIER_LABELS)

    tier_summary = (
        preds.groupby("tier", observed=True)
        .agg(n_customers=("customer_unique_id", "count"), repeat_rate=("y_true", "mean"), avg_predicted_proba=("y_proba", "mean"))
        .reset_index()
    )
    print(tier_summary)

    baseline_rate = preds["y_true"].mean()
    top_tier_rate = tier_summary.iloc[-1]["repeat_rate"]
    bottom_tier_rate = tier_summary.iloc[0]["repeat_rate"]
    lift = top_tier_rate / baseline_rate if baseline_rate > 0 else float("nan")
    print(f"Overall test repeat rate: {baseline_rate*100:.2f}%")
    print(f"Tier 5 (Very High) repeat rate: {top_tier_rate*100:.2f}% ({lift:.1f}x baseline)")
    print(f"Tier 1 (Very Low) repeat rate: {bottom_tier_rate*100:.2f}%")

    out_path = PROCESSED_DIR / "customer_segments.csv"
    preds.to_csv(out_path, index=False)
    print(f"Wrote {len(preds):,} scored customers to {out_path}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["segmentation"] = {
        "baseline_repeat_rate": round(float(baseline_rate), 4),
        "tier_5_repeat_rate": round(float(top_tier_rate), 4),
        "tier_5_lift_vs_baseline": round(float(lift), 2),
        "tiers": tier_summary.assign(
            repeat_rate=lambda d: d["repeat_rate"].round(4),
            avg_predicted_proba=lambda d: d["avg_predicted_proba"].round(4),
        ).to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    plot_tier_validation(tier_summary, baseline_rate)


def plot_tier_validation(tier_summary: pd.DataFrame, baseline_rate: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(tier_summary["tier"], tier_summary["repeat_rate"] * 100, color=COLOR_ACCENT)
    ax.axhline(baseline_rate * 100, color="#4A5B6E", linestyle="--", linewidth=1.2, label="Overall test-set average")
    ax.set_xlabel("Propensity tier (from CatBoost score, out-of-sample)")
    ax.set_ylabel("Actual repeat-purchase rate (%)")
    ax.set_title("Higher-scored tiers really do repeat-purchase more", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_tier_validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
