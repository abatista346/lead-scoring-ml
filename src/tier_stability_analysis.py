"""Quantify the instability that per-batch quantile tiering would have
introduced, versus the fixed probability thresholds used in
segment_customers.py.

Simulates two different scoring runs by splitting the test set into
customers whose first order fell in H1 (Jan-Jun) vs H2 (Jul-Dec) —
a real, non-arbitrary split standing in for "this month's batch" vs
"next month's batch" of leads to score. For each batch we compare:

  - OLD approach: recompute quantile tier edges independently on
    just that batch (pd.qcut per batch).
  - NEW approach: apply the same frozen fixed thresholds from
    segment_customers.py to both batches (pd.cut with fixed edges).

The OLD approach's edges drift between batches; the NEW approach's
edges are identical by construction. What matters commercially is how
many customers that drift actually reclassifies.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segment_customers import TIER_LABELS, compute_fixed_thresholds

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

COLOR_ACCENT = "#1D5FD6"
COLOR_MUTED = "#9AAABB"
COLOR_TEXT_SECONDARY = "#4A5B6E"

BOUNDARY_NAMES = ["1/2", "2/3", "3/4", "4/5"]


def load_data() -> pd.DataFrame:
    preds = pd.read_csv(PROCESSED_DIR / "catboost_test_predictions.csv")
    feats = pd.read_csv(PROCESSED_DIR / "model_features.csv")[["customer_unique_id", "purchase_month"]]
    df = preds.merge(feats, on="customer_unique_id", how="left")
    df["batch"] = np.where(df["purchase_month"] <= 6, "H1 (Jan-Jun)", "H2 (Jul-Dec)")
    return df


def main() -> None:
    df = load_data()

    fixed_edges = compute_fixed_thresholds(df["y_proba"])
    df["tier_fixed"] = pd.cut(df["y_proba"], bins=fixed_edges, labels=TIER_LABELS)

    batch_edges = {}
    for batch, group in df.groupby("batch"):
        _, edges = pd.qcut(group["y_proba"], 5, retbins=True, duplicates="drop")
        batch_edges[batch] = edges
    df["tier_quantile_perbatch"] = df.groupby("batch")["y_proba"].transform(
        lambda s: pd.qcut(s, 5, labels=TIER_LABELS, duplicates="drop")
    )

    mismatch = df["tier_fixed"].astype(str) != df["tier_quantile_perbatch"].astype(str)
    mismatch_rate = mismatch.mean()
    print(f"Customers whose tier differs between per-batch quantiles and fixed thresholds: "
          f"{mismatch.sum():,} / {len(df):,} ({mismatch_rate*100:.1f}%)")

    example = df[mismatch].iloc[(mismatch.sum() // 2)]
    print("Example — identical probability, different tier depending on batch composition:")
    print(example[["y_proba", "batch", "tier_fixed", "tier_quantile_perbatch"]])

    for batch, edges in batch_edges.items():
        print(f"{batch} quantile edges: {[round(e, 4) for e in edges]}")
    print(f"Fixed thresholds (frozen, same for every batch): {[round(e, 4) for e in fixed_edges if np.isfinite(e)]}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["tier_stability_analysis"] = {
        "n_customers": len(df),
        "reclassified_count": int(mismatch.sum()),
        "reclassified_pct": round(float(mismatch_rate) * 100, 2),
        "batch_quantile_edges": {b: [round(float(e), 4) for e in edges] for b, edges in batch_edges.items()},
        "fixed_edges": [round(float(e), 4) for e in fixed_edges if np.isfinite(e)],
    }
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Wrote stability analysis to {metrics_path}")

    plot_boundary_drift(batch_edges, fixed_edges)


def plot_boundary_drift(batch_edges: dict, fixed_edges: np.ndarray) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fixed_inner = [e for e in fixed_edges if np.isfinite(e)]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_positions = np.arange(len(BOUNDARY_NAMES))

    colors = {"H1 (Jan-Jun)": COLOR_MUTED, "H2 (Jul-Dec)": COLOR_TEXT_SECONDARY}
    for batch, edges in batch_edges.items():
        inner = list(edges[1:-1])
        ax.scatter(inner, y_positions, color=colors.get(batch, COLOR_MUTED), s=70, label=f"{batch} (recomputed per batch)", zorder=3)
    ax.scatter(fixed_inner, y_positions, color=COLOR_ACCENT, s=110, marker="D", label="Fixed threshold (frozen)", zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Tier {b} boundary" for b in BOUNDARY_NAMES])
    ax.set_xlabel("Probability cutoff")
    ax.set_title("Quantile boundaries drift between batches; fixed thresholds don't", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_tier_stability.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
