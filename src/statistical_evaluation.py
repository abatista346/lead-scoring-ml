"""Statistical evaluation of the propensity models, beyond point metrics.

A single ROC-AUC number on one test split does not say whether one model
is actually better than another, whether the predicted probabilities can
be trusted as probabilities, or whether the tier spread is larger than
sampling noise. This script answers those three questions:

1. Uncertainty — 95% bootstrap confidence intervals for ROC-AUC and
   PR-AUC, plus a *paired* bootstrap of the CatBoost − XGBoost difference
   (both models are scored on the same customers, so resampling customers
   jointly is the correct way to compare them).
2. Calibration — the models were trained with balanced class weights,
   which deliberately pushes predicted probabilities away from the true
   ~3% base rate. Probabilities are recalibrated with isotonic regression
   using 5-fold cross-fitting on the test predictions (each customer's
   calibrated score comes from a calibrator that never saw that
   customer), and Brier score / expected calibration error are compared
   before and after.
3. Tier significance — a two-proportion z-test of Tier 5 vs. Tier 1
   repeat rates, and Wilson 95% intervals for every tier's repeat rate.

Outputs:
- data/processed/catboost_test_predictions_calibrated.csv (used by
  business_impact.py)
- outputs/figures/12_calibration_curve.png
- outputs/figures/13_tier_rates_with_ci.png
- outputs/metrics.json -> "statistical_evaluation" section
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

N_BOOTSTRAP = 1000
RANDOM_STATE = 42
N_CALIBRATION_BINS = 10

COLOR_ACCENT = "#1D5FD6"
COLOR_TEXT_SECONDARY = "#4A5B6E"
COLOR_MUTED = "#9AAABB"
COLOR_GRID = "#D7E3F0"


def bootstrap_metrics(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired bootstrap: resample customers once, score both models on the same resample."""
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        auc_a, auc_b = roc_auc_score(yb, p_a[idx]), roc_auc_score(yb, p_b[idx])
        ap_a, ap_b = average_precision_score(yb, p_a[idx]), average_precision_score(yb, p_b[idx])
        rows.append((auc_a, auc_b, auc_a - auc_b, ap_a, ap_b, ap_a - ap_b))
    return dict(zip(["auc_a", "auc_b", "auc_diff", "ap_a", "ap_b", "ap_diff"], np.array(rows).T))


def ci(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def cross_fitted_isotonic(y: np.ndarray, p: np.ndarray, seed: int) -> np.ndarray:
    """Calibrate with out-of-fold isotonic regression so no customer calibrates itself."""
    calibrated = np.zeros_like(p, dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in folds.split(p.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p[fit_idx], y[fit_idx])
        calibrated[apply_idx] = iso.predict(p[apply_idx])
    return calibrated


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    return (
        pd.DataFrame({"y": y, "p": p, "bin": bins})
        .groupby("bin")
        .agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    )


def expected_calibration_error(table: pd.DataFrame) -> float:
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def main() -> None:
    cat = pd.read_csv(PROCESSED_DIR / "catboost_test_predictions.csv")
    xgb = pd.read_csv(PROCESSED_DIR / "xgboost_test_predictions.csv")
    merged = cat.merge(xgb[["customer_unique_id", "y_proba"]], on="customer_unique_id", suffixes=("_cat", "_xgb"))
    y = merged["y_true"].to_numpy()
    p_cat, p_xgb = merged["y_proba_cat"].to_numpy(), merged["y_proba_xgb"].to_numpy()

    # 1. Uncertainty
    boot = bootstrap_metrics(y, p_cat, p_xgb, N_BOOTSTRAP, RANDOM_STATE)
    p_value_auc = float(min(1.0, 2 * min((boot["auc_diff"] <= 0).mean(), (boot["auc_diff"] >= 0).mean())))
    uncertainty = {
        "n_bootstrap": int(len(boot["auc_a"])),
        "catboost_roc_auc_ci95": ci(boot["auc_a"]),
        "xgboost_roc_auc_ci95": ci(boot["auc_b"]),
        "catboost_minus_xgboost_roc_auc": round(float(np.mean(boot["auc_diff"])), 4),
        "catboost_minus_xgboost_roc_auc_ci95": ci(boot["auc_diff"]),
        "catboost_minus_xgboost_roc_auc_bootstrap_p": round(p_value_auc, 4),
        "catboost_pr_auc_ci95": ci(boot["ap_a"]),
        "xgboost_pr_auc_ci95": ci(boot["ap_b"]),
        "catboost_minus_xgboost_pr_auc_ci95": ci(boot["ap_diff"]),
    }
    print(json.dumps(uncertainty, indent=2))

    # 2. Calibration
    p_cal = cross_fitted_isotonic(y, p_cat, RANDOM_STATE)
    raw_table = reliability_table(y, p_cat, N_CALIBRATION_BINS)
    cal_table = reliability_table(y, p_cal, N_CALIBRATION_BINS)
    calibration = {
        "method": "isotonic regression, 5-fold cross-fitted on test predictions",
        "brier_raw": round(float(brier_score_loss(y, p_cat)), 5),
        "brier_calibrated": round(float(brier_score_loss(y, p_cal)), 5),
        "brier_constant_base_rate": round(float(brier_score_loss(y, np.full_like(p_cat, y.mean()))), 5),
        "ece_raw": round(expected_calibration_error(raw_table), 4),
        "ece_calibrated": round(expected_calibration_error(cal_table), 4),
        "mean_predicted_raw": round(float(p_cat.mean()), 4),
        "mean_predicted_calibrated": round(float(p_cal.mean()), 4),
        "observed_rate": round(float(y.mean()), 4),
        "roc_auc_calibrated": round(float(roc_auc_score(y, p_cal)), 4),
    }
    print(json.dumps(calibration, indent=2))

    out = merged[["customer_unique_id", "y_true"]].assign(y_proba=p_cat, y_proba_calibrated=p_cal)
    out.to_csv(PROCESSED_DIR / "catboost_test_predictions_calibrated.csv", index=False)

    # 3. Tier significance (tiers from segment_customers.py)
    segments = pd.read_csv(PROCESSED_DIR / "customer_segments.csv")
    tiers = segments.groupby("tier").agg(repeats=("y_true", "sum"), n=("y_true", "size")).sort_index()
    tiers["rate"] = tiers["repeats"] / tiers["n"]
    lo, hi = proportion_confint(tiers["repeats"], tiers["n"], alpha=0.05, method="wilson")
    tiers["ci_low"], tiers["ci_high"] = lo, hi
    top, bottom = tiers.iloc[-1], tiers.iloc[0]
    z_stat, z_p = proportions_ztest([top["repeats"], bottom["repeats"]], [top["n"], bottom["n"]], alternative="larger")
    tier_test = {
        "test": "two-proportion z-test, Tier 5 > Tier 1 (one-sided)",
        "tier_5_rate": round(float(top["rate"]), 4),
        "tier_1_rate": round(float(bottom["rate"]), 4),
        "z": round(float(z_stat), 2),
        "p_value": float(f"{z_p:.2e}"),
        "tiers": [
            {"tier": t, "n": int(r["n"]), "repeat_rate": round(float(r["rate"]), 4),
             "ci95": [round(float(r["ci_low"]), 4), round(float(r["ci_high"]), 4)]}
            for t, r in tiers.iterrows()
        ],
    }
    print(json.dumps(tier_test, indent=2))

    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["statistical_evaluation"] = {"uncertainty": uncertainty, "calibration": calibration, "tier_significance": tier_test}
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    plot_calibration(raw_table, cal_table, y.mean())
    plot_tier_ci(tiers, y.mean())


def plot_calibration(raw: pd.DataFrame, cal: pd.DataFrame, base_rate: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, table, title in [
        (axes[0], raw, "Raw CatBoost scores (class-weighted)"),
        (axes[1], cal, "After isotonic calibration (cross-fitted)"),
    ]:
        lim = max(table["mean_predicted"].max(), table["observed_rate"].max()) * 1.1
        ax.plot([0, lim], [0, lim], color=COLOR_GRID, linestyle="--", linewidth=1, label="Perfect calibration")
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", color=COLOR_ACCENT, label="Model (score deciles)")
        ax.axhline(base_rate, color=COLOR_MUTED, linewidth=1, linestyle=":", label="Overall repeat rate")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, max(table["observed_rate"].max() * 1.3, base_rate * 2))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed repeat rate")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Reliability of predicted probabilities, before and after calibration", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_calibration_curve.png", dpi=150)
    plt.close(fig)


def plot_tier_ci(tiers: pd.DataFrame, base_rate: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tiers))
    rate = tiers["rate"].to_numpy() * 100
    err = np.vstack([rate - tiers["ci_low"].to_numpy() * 100, tiers["ci_high"].to_numpy() * 100 - rate])
    ax.bar(x, rate, color=COLOR_ACCENT)
    ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=COLOR_TEXT_SECONDARY, capsize=4, linewidth=1.2)
    ax.axhline(base_rate * 100, color=COLOR_TEXT_SECONDARY, linestyle="--", linewidth=1.2, label="Overall test-set average")
    ax.set_xticks(x, tiers.index)
    ax.set_ylabel("Actual repeat-purchase rate (%)")
    ax.set_xlabel("Propensity tier (95% Wilson intervals)")
    ax.set_title("Tier repeat rates with sampling uncertainty", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_tier_rates_with_ci.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
