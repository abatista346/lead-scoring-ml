"""Translate the propensity score into a retention-campaign decision.

Question a CRM manager actually asks: "If I run a win-back campaign after
a customer's first order, who should I contact, and is it worth it?"

The model predicts *propensity* (who is likely to buy again), not
*uplift* (who buys again *because* they were contacted). Without a
randomized experiment, the effect of contacting a customer is unknown,
so it is treated here as an explicit, configurable assumption and the
recommendation is stress-tested across a range of values. The natural
next step — measuring that effect with an A/B test — is spelled out in
the generated recommendation.

Economics per contacted customer i:
    expected incremental profit = p_i * RELATIVE_UPLIFT * margin_per_repeat - CONTACT_COST
where p_i is the *calibrated* probability (from statistical_evaluation.py).
Raw class-weighted scores sit far above the ~3% true base rate and
would overstate the value of every contact.

Outputs:
- outputs/figures/14_cumulative_gains.png
- outputs/figures/15_campaign_profit.png
- outputs/figures/16_profit_sensitivity.png
- outputs/metrics.json -> "business_impact" section
- README.md section between the BUSINESS_IMPACT markers, regenerated
  from the numbers above on every run
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
README_PATH = ROOT / "README.md"
START_MARKER = "<!-- BUSINESS_IMPACT:START -->"
END_MARKER = "<!-- BUSINESS_IMPACT:END -->"

# --- Campaign assumptions (edit these; everything downstream recomputes) ---
CONTACT_COST = 0.20          # R$ per contacted customer (email/push send + ops; discount cost is netted into the margin rate)
GROSS_MARGIN_RATE = 0.30     # share of a repeat order's value kept as gross margin
RELATIVE_UPLIFT = 0.15       # contacting raises a customer's repeat probability by 15% (relative)
UPLIFT_GRID = [0.05, 0.10, 0.15, 0.25, 0.40]
COST_GRID = [0.05, 0.10, 0.20, 0.40, 0.80]
SCALE_CUSTOMERS_PER_MONTH = 10_000  # illustrative volume of new first-time buyers

COLOR_ACCENT = "#1D5FD6"
COLOR_TEXT_SECONDARY = "#4A5B6E"
COLOR_MUTED = "#9AAABB"
COLOR_GRID = "#D7E3F0"


def gains_table(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    ranked = df.sort_values("y_proba_calibrated", ascending=False).reset_index(drop=True)
    ranked["decile"] = pd.qcut(ranked.index, n_deciles, labels=range(1, n_deciles + 1))
    table = ranked.groupby("decile", observed=True).agg(n=("y_true", "size"), repeats=("y_true", "sum"))
    table["repeat_rate"] = table["repeats"] / table["n"]
    table["cum_share_contacted"] = table["n"].cumsum() / table["n"].sum()
    table["cum_share_repeaters_captured"] = table["repeats"].cumsum() / table["repeats"].sum()
    table["lift"] = table["repeat_rate"] / ranked["y_true"].mean()
    return table


def profit_curve(p_sorted: np.ndarray, margin: float, uplift: float, cost: float) -> np.ndarray:
    """Cumulative expected incremental profit when contacting the top-k customers, k = 0..n."""
    per_customer = p_sorted * uplift * margin - cost
    return np.concatenate([[0.0], np.cumsum(per_customer)])


def best_policy(p_sorted: np.ndarray, margin: float, uplift: float, cost: float) -> tuple[int, float]:
    curve = profit_curve(p_sorted, margin, uplift, cost)
    k = int(np.argmax(curve))
    return k, float(curve[k])


def main() -> None:
    preds = pd.read_csv(PROCESSED_DIR / "catboost_test_predictions_calibrated.csv")
    feats = pd.read_csv(PROCESSED_DIR / "model_features.csv", usecols=["customer_unique_id", "payment_value", "target_repeat_purchase", "split"])

    # Value of a repeat order: proxied by the median first-order value of
    # customers who did repeat (training split only, to keep test untouched).
    train_repeaters = feats[(feats["split"] == "train") & (feats["target_repeat_purchase"] == 1)]
    repeat_order_value = float(train_repeaters["payment_value"].median())
    margin_per_repeat = repeat_order_value * GROSS_MARGIN_RATE

    n = len(preds)
    base_rate = float(preds["y_true"].mean())
    p_sorted = np.sort(preds["y_proba_calibrated"].to_numpy())[::-1]

    gains = gains_table(preds)
    top20_capture = float(gains.loc[gains["cum_share_contacted"] <= 0.2001, "cum_share_repeaters_captured"].iloc[-1])

    k_best, profit_best = best_policy(p_sorted, margin_per_repeat, RELATIVE_UPLIFT, CONTACT_COST)
    profit_all = float(profit_curve(p_sorted, margin_per_repeat, RELATIVE_UPLIFT, CONTACT_COST)[-1])
    breakeven_p = CONTACT_COST / (RELATIVE_UPLIFT * margin_per_repeat)

    sensitivity = pd.DataFrame(index=pd.Index(UPLIFT_GRID, name="relative_uplift"), columns=pd.Index(COST_GRID, name="contact_cost"), dtype=float)
    share_grid = sensitivity.copy()
    for u in UPLIFT_GRID:
        for c in COST_GRID:
            k, prof = best_policy(p_sorted, margin_per_repeat, u, c)
            sensitivity.loc[u, c] = prof / n * SCALE_CUSTOMERS_PER_MONTH
            share_grid.loc[u, c] = k / n

    result = {
        "assumptions": {
            "contact_cost_brl": CONTACT_COST,
            "gross_margin_rate": GROSS_MARGIN_RATE,
            "relative_uplift": RELATIVE_UPLIFT,
            "repeat_order_value_brl_median_train": round(repeat_order_value, 2),
            "margin_per_repeat_brl": round(margin_per_repeat, 2),
        },
        "base_repeat_rate": round(base_rate, 4),
        "top_decile_lift": round(float(gains["lift"].iloc[0]), 2),
        "top_20pct_share_of_repeaters_captured": round(top20_capture, 4),
        "breakeven_calibrated_probability": round(breakeven_p, 4),
        "optimal_share_contacted": round(k_best / n, 4),
        "profit_optimal_per_10k_customers_brl": round(profit_best / n * SCALE_CUSTOMERS_PER_MONTH, 0),
        "profit_contact_everyone_per_10k_customers_brl": round(profit_all / n * SCALE_CUSTOMERS_PER_MONTH, 0),
        "sensitivity_profit_per_10k_brl": {str(u): {str(c): round(float(v), 0) for c, v in row.items()} for u, row in sensitivity.iterrows()},
        "sensitivity_share_contacted": {str(u): {str(c): round(float(v), 3) for c, v in row.items()} for u, row in share_grid.iterrows()},
        "gains_by_decile": gains.reset_index().assign(decile=lambda d: d["decile"].astype(int)).round(4).to_dict(orient="records"),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "gains_by_decile"}, indent=2))

    metrics_path = OUTPUTS_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics["business_impact"] = result
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    plot_gains(gains)
    plot_profit(p_sorted, margin_per_repeat, k_best)
    plot_sensitivity(sensitivity, share_grid)
    write_readme_section(result, all_metrics.get("statistical_evaluation"), gains, sensitivity)


def plot_gains(gains: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([[0], gains["cum_share_contacted"].to_numpy()]) * 100
    y = np.concatenate([[0], gains["cum_share_repeaters_captured"].to_numpy()]) * 100
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([0, 100], [0, 100], color=COLOR_GRID, linestyle="--", linewidth=1, label="Random targeting")
    ax.plot(x, y, marker="o", color=COLOR_ACCENT, label="Model-ranked targeting")
    ax.set_xlabel("% of first-time buyers contacted (highest score first)")
    ax.set_ylabel("% of all repeat buyers reached")
    ax.set_title("Cumulative gains: how many future repeat buyers each budget reaches", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_cumulative_gains.png", dpi=150)
    plt.close(fig)


def plot_profit(p_sorted: np.ndarray, margin: float, k_best: int) -> None:
    n = len(p_sorted)
    share = np.arange(n + 1) / n * 100
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for uplift, color, width in [(0.05, COLOR_MUTED, 1.2), (RELATIVE_UPLIFT, COLOR_ACCENT, 2.2), (0.40, COLOR_TEXT_SECONDARY, 1.2)]:
        curve = profit_curve(p_sorted, margin, uplift, CONTACT_COST) / n * SCALE_CUSTOMERS_PER_MONTH
        ax.plot(share, curve, color=color, linewidth=width, label=f"Uplift {uplift:.0%}" + (" (base case)" if uplift == RELATIVE_UPLIFT else ""))
    ax.axhline(0, color=COLOR_GRID, linewidth=1)
    ax.axvline(k_best / n * 100, color=COLOR_ACCENT, linestyle=":", linewidth=1.2, label="Base-case optimum")
    ax.set_xlabel("% of first-time buyers contacted (highest score first)")
    ax.set_ylabel(f"Expected incremental profit per {SCALE_CUSTOMERS_PER_MONTH:,} buyers (R$)")
    ax.set_title(f"Campaign profit by targeting depth (contact cost R${CONTACT_COST:.2f})", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_campaign_profit.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, share_grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = sensitivity.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(sensitivity.columns)), [f"R${c:.2f}" for c in sensitivity.columns])
    ax.set_yticks(range(len(sensitivity.index)), [f"{u:.0%}" for u in sensitivity.index])
    ax.set_xlabel("Cost per contact")
    ax.set_ylabel("Relative uplift from contact (unknown — needs A/B test)")
    threshold = np.nanmax(values) * 0.6 if np.nanmax(values) > 0 else 1
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            share = share_grid.iloc[i, j]
            if share == 0 or values[i, j] < 1:
                label = "skip\ncampaign"
            else:
                share_txt = "<1%" if share < 0.005 else f"{share:.0%}"
                label = f"R${values[i, j]:,.0f}\ncontact {share_txt}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if values[i, j] > threshold else "#1A2433")
    ax.set_title(f"Best achievable profit per {SCALE_CUSTOMERS_PER_MONTH:,} buyers, by assumption", loc="left", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="R$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_profit_sensitivity.png", dpi=150)
    plt.close(fig)


def write_readme_section(r: dict, stats: dict | None, gains: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    a = r["assumptions"]
    lines = [START_MARKER, "", "## Statistical Rigor", ""]
    if stats:
        u, c, t = stats["uncertainty"], stats["calibration"], stats["tier_significance"]
        diff_ci = u["catboost_minus_xgboost_roc_auc_ci95"]
        same = diff_ci[0] <= 0 <= diff_ci[1]
        lines += [
            "Point metrics on one split hide uncertainty, so every headline number is re-examined "
            f"with {u['n_bootstrap']:,} paired bootstrap resamples of the test set "
            "(`src/statistical_evaluation.py`).",
            "",
            "| Question | Result |",
            "|---|---|",
            f"| CatBoost ROC-AUC, 95% CI | {u['catboost_roc_auc_ci95'][0]:.3f} – {u['catboost_roc_auc_ci95'][1]:.3f} |",
            f"| XGBoost ROC-AUC, 95% CI | {u['xgboost_roc_auc_ci95'][0]:.3f} – {u['xgboost_roc_auc_ci95'][1]:.3f} |",
            f"| CatBoost − XGBoost ROC-AUC, 95% CI | {diff_ci[0]:+.3f} to {diff_ci[1]:+.3f} (bootstrap p = {u['catboost_minus_xgboost_roc_auc_bootstrap_p']:.2f}) |",
            f"| Brier score: raw → calibrated (constant base rate) | {c['brier_raw']:.4f} → {c['brier_calibrated']:.4f} ({c['brier_constant_base_rate']:.4f}) |",
            f"| Expected calibration error: raw → calibrated | {c['ece_raw']:.3f} → {c['ece_calibrated']:.3f} |",
            f"| Tier 5 vs Tier 1 repeat rate (one-sided z-test) | {t['tier_5_rate']:.2%} vs {t['tier_1_rate']:.2%}, z = {t['z']:.1f}, p = {t['p_value']:.1e} |",
            "",
            ("- **The CatBoost \"win\" is not statistically distinguishable from XGBoost** — the confidence interval of the difference includes zero. "
             "Either library is a defensible choice; the decision should rest on operational factors (native categorical handling, inference cost), not this AUC gap."
             if same else
             "- **CatBoost's advantage over XGBoost holds under resampling** — the confidence interval of the difference excludes zero."),
            f"- **Raw scores are not probabilities.** Balanced class weights push the average predicted score to {c['mean_predicted_raw']:.1%} against a true repeat rate of {c['observed_rate']:.1%}. "
            "Cross-fitted isotonic calibration (each customer calibrated by a model that never saw it) brings them back in line, which is required before any expected-value calculation below.",
            (f"- **The tier spread is real**: Tier 5 vs Tier 1 is significant at p = {t['p_value']:.1e}"
             + (", and the Wilson intervals of the extreme tiers do not overlap (figure below)."
                if t["tiers"][0]["ci95"][1] < t["tiers"][-1]["ci95"][0] else
                ", although the Wilson intervals of the extreme tiers overlap (figure below).")
             if t["p_value"] < 0.05 else
             f"- **The tier spread is not statistically significant** (p = {t['p_value']:.2f}) — tiers should not drive different treatment until more data is available."),
            "",
            "![Calibration curve](outputs/figures/12_calibration_curve.png)",
            "![Tier rates with confidence intervals](outputs/figures/13_tier_rates_with_ci.png)",
            "",
        ]
    lines += [
        "## Business Recommendation",
        "",
        "**Decision framed:** after a customer's first order, should the CRM team send a win-back offer, and to whom?",
        "",
        f"**Targeting efficiency.** Contacting the top 20% of first-time buyers by score reaches **{r['top_20pct_share_of_repeaters_captured']:.0%} of all future repeat buyers**, "
        f"and the top decile repeats at **{r['top_decile_lift']:.1f}x** the base rate of {r['base_repeat_rate']:.1%}.",
        "",
        "![Cumulative gains](outputs/figures/14_cumulative_gains.png)",
        "",
        "**Campaign economics (`src/business_impact.py`).** Assumptions are explicit and editable at the top of the script:",
        "",
        f"- Value of a repeat order: R${a['repeat_order_value_brl_median_train']:.2f} (median first-order value of training-set repeat buyers) × {a['gross_margin_rate']:.0%} gross margin = **R${a['margin_per_repeat_brl']:.2f}** per repeat order",
        f"- Cost per contact: **R${a['contact_cost_brl']:.2f}**",
        f"- Effect of contact: **+{a['relative_uplift']:.0%} relative** increase in repeat probability — *an assumption*, because this model measures propensity, not causal uplift",
        "",
        f"Under these assumptions a contact pays for itself only when a customer's calibrated repeat probability exceeds **{r['breakeven_calibrated_probability']:.1%}**. "
        + (f"The profit-maximizing policy is to contact the top **{r['optimal_share_contacted']:.0%}** of first-time buyers, for an expected **R${r['profit_optimal_per_10k_customers_brl']:,.0f}** "
           f"per 10,000 buyers, versus **R${r['profit_contact_everyone_per_10k_customers_brl']:,.0f}** for contacting everyone."
           if r["optimal_share_contacted"] > 0 else
           "No segment clears that bar, so the profit-maximizing decision under these assumptions is **not to run a paid campaign** — a legitimate outcome of the analysis."),
        "",
        "![Campaign profit curve](outputs/figures/15_campaign_profit.png)",
        "",
        "**How robust is that?** The answer depends far more on the unknown uplift than on the model. The grid below shows the best achievable profit and the share of buyers to contact for each combination:",
        "",
        "![Profit sensitivity](outputs/figures/16_profit_sensitivity.png)",
        "",
        "**Recommendation.**",
        "",
        "1. Use the score to *rank*, and the calibrated probability with the break-even rule above to decide *how deep* to go — never the raw class-weighted score.",
        "2. Before scaling, run a randomized holdout: within the targeted tiers, withhold the offer from a random control group and measure the true incremental repeat rate. That converts the single biggest assumption here into a measured number and tells you whether the campaign creates repeat purchases or just discounts ones that would have happened anyway.",
        "3. Re-run this script with the measured uplift and actual contact cost to set the final targeting depth.",
        "",
        END_MARKER,
    ]
    section = "\n".join(lines)
    readme = README_PATH.read_text()
    if START_MARKER in readme and END_MARKER in readme:
        before = readme.split(START_MARKER)[0]
        after = readme.split(END_MARKER)[1]
        readme = before + section + after
    else:
        anchor = "## Architecture"
        readme = readme.replace(anchor, section + "\n\n" + anchor, 1) if anchor in readme else readme + "\n\n" + section + "\n"
    README_PATH.write_text(readme)
    print(f"Updated {README_PATH.name} business-impact section")


if __name__ == "__main__":
    main()
