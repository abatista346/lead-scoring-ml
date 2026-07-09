"""Segment customers into 5 propensity tiers using the CatBoost
out-of-sample predictions (the winning model from train_catboost.py).

Tier boundaries are FIXED absolute probability cutoffs, computed once
from this reference (test) population and then frozen — not
percentile/quantile cutoffs recomputed on whatever batch happens to be
scored. Quantile tiering ("top 20% of this batch") is unstable: the same
customer, with the exact same predicted probability, can land in a
different tier purely because of who else was in that scoring batch.
Fixed thresholds keep "Tier 5" meaning the same probability range every
time, which is what makes tier-based business rules (e.g. "call Tier 5
within 24h") actually usable month over month. See
tier_stability_analysis.py for a concrete before/after measurement of
this instability.

Deliberately scores only the held-out test set rather than refitting on
the full population — every repeat-purchase rate reported per tier is
therefore genuinely out-of-sample, not inflated by the model having
seen those customers during training. In production you'd retrain on
all available data before scoring new/current leads, then apply these
same frozen thresholds.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

COLOR_ACCENT = "#1D5FD6"
TIER_LABELS = ["1 - Very Low", "2 - Low", "3 - Medium", "4 - High", "5 - Very High"]


def compute_fixed_thresholds(reference_proba: pd.Series) -> np.ndarray:
    """Freeze quintile edges from a reference population as fixed cutoffs.

    The inner edges come from this reference population's distribution,
    but once computed they are constants — reused as-is on every future
    scoring batch instead of being recomputed per batch.
    """
    _, edges = pd.qcut(reference_proba, 5, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def main() -> None:
    preds = pd.read_csv(PROCESSED_DIR / "catboost_test_predictions.csv")

    fixed_edges = compute_fixed_thresholds(preds["y_proba"])
    preds["tier"] = pd.cut(preds["y_proba"], bins=fixed_edges, labels=TIER_LABELS)

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
    print(f"Frozen tier thresholds (probability): {[round(e, 4) for e in fixed_edges]}")

    out_path = PROCESSED_DIR / "customer_segments.csv"
    preds.to_csv(out_path, index=False)
    print(f"Wrote {len(preds):,} scored customers to {out_path}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["segmentation"] = {
        "method": "fixed_probability_thresholds",
        "fixed_thresholds": [None if not np.isfinite(e) else round(float(e), 4) for e in fixed_edges],
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
    ax.set_xlabel("Propensity tier (fixed probability thresholds, out-of-sample)")
    ax.set_ylabel("Actual repeat-purchase rate (%)")
    ax.set_title("Higher-scored tiers really do repeat-purchase more", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_tier_validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
