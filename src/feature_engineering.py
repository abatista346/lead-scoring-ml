"""Impute missing values, encode categoricals, and create a stratified
train/test split for the customer-level propensity model.

The split is saved as a column on the output file (rather than separate
train/test CSVs) so every downstream script — baseline, XGBoost, CatBoost,
segmentation — uses the exact same split for a fair comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_CATEGORIES = 15
TOP_STATES = 10


def impute_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    item_cols = ["n_items", "n_distinct_sellers", "n_distinct_categories", "items_price", "freight_value", "freight_ratio"]
    for col in item_cols:
        df[col] = df[col].fillna(df[col].median())

    df["primary_category"] = df["primary_category"].fillna("unknown")
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["payment_value"] = df["payment_value"].fillna(df["payment_value"].median())
    df["payment_installments"] = df["payment_installments"].fillna(df["payment_installments"].median())

    df["has_review"] = df["review_score"].notna().astype(int)
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())

    df["is_delivered"] = df["delivery_days"].notna().astype(int)
    df["delivery_days"] = df["delivery_days"].fillna(df["delivery_days"].median())

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    top_categories = df["primary_category"].value_counts().head(TOP_CATEGORIES).index
    df["primary_category_grouped"] = df["primary_category"].where(df["primary_category"].isin(top_categories), "other")

    top_states = df["customer_state"].value_counts().head(TOP_STATES).index
    df["customer_state_grouped"] = df["customer_state"].where(df["customer_state"].isin(top_states), "other")

    df = pd.get_dummies(
        df,
        columns=["primary_category_grouped", "customer_state_grouped", "payment_type"],
        prefix=["category", "state", "payment"],
        dtype=int,
    )
    return df.drop(columns=["primary_category", "customer_state"])


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "customer_first_order_features.csv")
    df = impute_and_flag(df)
    df = encode_categoricals(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target_repeat_purchase"],
    )
    df["split"] = np.where(df.index.isin(train_idx), "train", "test")

    out_path = PROCESSED_DIR / "model_features.csv"
    df.to_csv(out_path, index=False)

    train_rate = df.loc[df["split"] == "train", "target_repeat_purchase"].mean()
    test_rate = df.loc[df["split"] == "test", "target_repeat_purchase"].mean()
    print(f"Wrote {len(df):,} rows, {df.shape[1]} columns to {out_path}")
    print(f"Train: {len(train_idx):,} rows, repeat rate {train_rate*100:.2f}%")
    print(f"Test: {len(test_idx):,} rows, repeat rate {test_rate*100:.2f}%")


if __name__ == "__main__":
    main()
