"""Tests for the segmentation, statistical evaluation and campaign economics."""

import numpy as np
import pandas as pd
import pytest

from business_impact import best_policy, gains_table, profit_curve
from segment_customers import compute_fixed_thresholds
from statistical_evaluation import (
    bootstrap_metrics,
    cross_fitted_isotonic,
    expected_calibration_error,
    reliability_table,
)


@pytest.fixture
def scored() -> pd.DataFrame:
    """Imbalanced labels (~3%) with an informative but inflated score, like class-weighted output."""
    rng = np.random.default_rng(42)
    n = 20_000
    signal = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-3.6 + 0.6 * signal))))
    raw = 0.2 + 0.6 / (1 + np.exp(-(signal + rng.normal(0, 0.8, n))))
    return pd.DataFrame({"y_true": y, "y_proba": raw})


def test_fixed_thresholds_are_frozen_quintiles(scored):
    edges = compute_fixed_thresholds(scored["y_proba"])
    assert len(edges) == 6 and np.isinf(edges[0]) and np.isinf(edges[-1])
    tiers = pd.cut(scored["y_proba"], bins=edges, labels=False)
    assert tiers.value_counts(normalize=True).between(0.19, 0.21).all()


def test_fixed_thresholds_do_not_depend_on_new_batch(scored):
    edges = compute_fixed_thresholds(scored["y_proba"])
    customer = 0.5
    small_batch = pd.Series([customer, 0.21, 0.22, 0.23])
    tier_alone = pd.cut(pd.Series([customer]), bins=edges, labels=False).iloc[0]
    tier_in_batch = pd.cut(small_batch, bins=edges, labels=False).iloc[0]
    assert tier_alone == tier_in_batch


def test_calibration_fixes_inflated_scores(scored):
    y, p = scored["y_true"].to_numpy(), scored["y_proba"].to_numpy()
    calibrated = cross_fitted_isotonic(y, p, seed=0)
    raw_ece = expected_calibration_error(reliability_table(y, p, 10))
    cal_ece = expected_calibration_error(reliability_table(y, calibrated, 10))
    assert raw_ece > 0.2
    assert cal_ece < 0.01
    assert calibrated.mean() == pytest.approx(y.mean(), abs=0.005)
    assert calibrated.min() >= 0 and calibrated.max() <= 1


def test_paired_bootstrap_detects_no_difference_for_identical_models(scored):
    y, p = scored["y_true"].to_numpy(), scored["y_proba"].to_numpy()
    boot = bootstrap_metrics(y, p, p, n_boot=50, seed=0)
    assert np.allclose(boot["auc_diff"], 0)


def test_paired_bootstrap_detects_a_better_model(scored):
    y, p = scored["y_true"].to_numpy(), scored["y_proba"].to_numpy()
    noise = np.random.default_rng(1).uniform(size=len(y))
    boot = bootstrap_metrics(y, p, noise, n_boot=100, seed=0)
    assert np.percentile(boot["auc_diff"], 2.5) > 0


def test_profit_curve_and_breakeven():
    p_sorted = np.array([0.10, 0.05, 0.02, 0.01])
    margin, uplift, cost = 100.0, 0.5, 2.0  # breakeven probability = 2 / (0.5 * 100) = 4%
    curve = profit_curve(p_sorted, margin, uplift, cost)
    assert curve[0] == 0 and len(curve) == len(p_sorted) + 1
    k, profit = best_policy(p_sorted, margin, uplift, cost)
    assert k == 2  # only the two customers above 4% are worth contacting
    assert profit == pytest.approx((0.10 * 50 - 2) + (0.05 * 50 - 2))


def test_no_campaign_when_contact_never_pays():
    k, profit = best_policy(np.array([0.01, 0.005]), margin=10.0, uplift=0.1, cost=5.0)
    assert k == 0 and profit == 0


def test_gains_table_captures_all_repeaters(scored):
    df = scored.assign(y_proba_calibrated=scored["y_proba"])
    gains = gains_table(df)
    assert gains["cum_share_repeaters_captured"].iloc[-1] == pytest.approx(1.0)
    assert gains["cum_share_contacted"].iloc[-1] == pytest.approx(1.0)
    assert gains["lift"].iloc[0] > 1  # the informative score ranks repeaters first
