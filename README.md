# Lead Scoring — Customer Repeat-Purchase Propensity

> ML lead scoring pipeline that predicts a customer's propensity to buy again from their first-order profile, comparing XGBoost vs. CatBoost and segmenting customers into 5 actionable propensity tiers.

## Problem

Sales and CRM teams need to prioritize which customers are worth pursuing for retention or upsell campaigns. This project builds a propensity score from a customer's very first purchase — before any repeat behavior is observed — and turns it into 5 actionable tiers a marketing team could act on directly.

## Data Source

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — the same dataset used in the companion [retail-sales-forecasting](https://github.com/abatista346/retail-sales-forecasting) repo, reframed here around individual customer behavior instead of aggregate demand. Data © Olist, released publicly on Kaggle under its dataset terms.

Not included in this repository (raw CSVs are gitignored). See [Setup](#setup) to fetch it yourself.

## Solution

This dataset has no traditional "leads" — every customer already completed a purchase. The propensity target is therefore defined as: **will this customer place a second order?**, predicted using only information available from their **first order** (so nothing about the second order leaks into the features).

1. **Ingestion** (`src/ingest.py`) — pulls the dataset via `kagglehub`.
2. **Preprocessing** (`src/preprocessing.py`) — builds one row per unique customer from their first order only (basket size, category, payment, delivery, review), with a `target_repeat_purchase` flag from whether they placed a second non-canceled order.
3. **EDA** (`notebooks/eda_storytelling.ipynb`) — checks which first-order signals actually predict repeat behavior.
4. **Feature engineering** (`src/feature_engineering.py`) — imputation, one-hot encoding, and a single stratified 80/20 train/test split reused by every model.
5. **Baselines** (`src/train_baseline.py`) — majority-class and class-weighted logistic regression.
6. **XGBoost** (`src/train_xgboost.py`) and **CatBoost** (`src/train_catboost.py`) — gradient boosting classifiers on the identical feature set, for a fair library comparison.
7. **Segmentation** (`src/segment_customers.py`) — buckets the winning model's out-of-sample scores into 5 propensity tiers.
8. **Evaluation** (`src/evaluate_models.py`) — ROC/PR curve comparison and feature importance.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, CatBoost, matplotlib, Jupyter, kagglehub

## Key Findings (EDA)

![Class imbalance](outputs/figures/01_class_imbalance.png)

- **Severe class imbalance**: only 3.04% of customers (2,888 of 94,990) ever place a second order — accuracy is a useless metric here; the majority-class baseline hits 97% accuracy while doing nothing.
- **Basket size is the strongest single signal**: customers who bought 4+ items in their first order repeat at roughly double the rate of single-item buyers.
- **Category effects are real but modest**: furniture/home categories retain noticeably better than electronics — a ~2.5x spread.
- **No single feature is a strong standalone predictor** — even review score is weak and non-monotonic, motivating a multivariate model over simple business rules.

See the [EDA notebook](notebooks/eda_storytelling.ipynb) for the full analysis.

## Model Results

Evaluated on a held-out, stratified 20% test set (18,998 customers, same 3.04% positive rate):

| Model | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| Majority-class | 0.500 | 0.030 | 97.0% |
| Logistic regression (class-weighted) | 0.613 | 0.047 | 60.1% |
| XGBoost | 0.640 | 0.054 | 68.5% |
| **CatBoost** | **0.642** | **0.054** | 65.4% |

![ROC curves](outputs/figures/08_roc_curves.png)
![Precision-recall curves](outputs/figures/09_pr_curves.png)

**CatBoost narrowly wins**, so it's the model used for segmentation below. The gap between XGBoost and CatBoost is small enough that, honestly, algorithm choice matters less here than getting the imbalance handling and feature set right — both boosted models clear logistic regression by a similar margin.

Top drivers of the CatBoost score: `freight_value`, `payment_installments`, `purchase_month`, and `delivery_days` — shipping cost and payment behavior matter more than product category.

![Feature importance](outputs/figures/10_feature_importance.png)

## Customer Segmentation

Even with a modest 0.642 ROC-AUC, the score **rank-orders customers well** — which is what actually matters for prioritization. Customers are split into 5 quintile tiers by predicted propensity, scored purely on the held-out test set so every number below is genuinely out-of-sample:

| Tier | Customers | Actual repeat rate | vs. baseline |
|---|---|---|---|
| 1 – Very Low | 3,800 | 1.29% | 0.4x |
| 2 – Low | 3,799 | 2.05% | 0.7x |
| 3 – Medium | 3,800 | 2.82% | 0.9x |
| 4 – High | 3,799 | 3.66% | 1.2x |
| 5 – Very High | 3,800 | **5.39%** | **1.8x** |

![Tier validation](outputs/figures/07_tier_validation.png)

Tier 5 customers repeat-purchase at 1.8x the overall rate, and 4.2x the rate of Tier 1 — a real, actionable spread a retention team could use to prioritize outreach, even though no individual first-order signal is strongly predictive on its own.

## Architecture

```
data/raw/            <- gitignored, populated by src/ingest.py
data/processed/       <- gitignored, populated by preprocessing + feature engineering
src/
  ingest.py            <- kagglehub download
  preprocessing.py     <- first-order customer table + repeat-purchase target
  feature_engineering.py <- imputation, encoding, train/test split
  train_baseline.py    <- majority-class + logistic regression
  train_xgboost.py      <- XGBoost classifier
  train_catboost.py     <- CatBoost classifier
  segment_customers.py <- 5-tier propensity segmentation
  evaluate_models.py    <- ROC/PR curves, feature importance
notebooks/
  eda_storytelling.ipynb
outputs/
  figures/             <- exported charts
  metrics.json          <- every model's metrics + tier summary
```

## Setup

1. `pip install -r requirements.txt`
2. `python src/ingest.py` (public dataset — no Kaggle API token required)
3. `python src/preprocessing.py`
4. `python src/feature_engineering.py`
5. `python src/train_baseline.py`
6. `python src/train_xgboost.py`
7. `python src/train_catboost.py`
8. `python src/segment_customers.py`
9. `python src/evaluate_models.py`
10. Optionally, open `notebooks/eda_storytelling.ipynb` for the full visual EDA

## What I'd Improve Next

1. Bring in `customer_zip_code_prefix` / geolocation data as a proper distance or region feature — geography was only used at the state level here and is likely under-exploited.
2. Try a monotonic-constrained or calibrated model (e.g. isotonic calibration) so the predicted probabilities themselves are trustworthy for tier cutoffs, not just the rank ordering.
