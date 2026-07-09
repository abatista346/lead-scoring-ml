"""Build a customer-level table from the Olist data: one row per unique
customer, features describing only their FIRST order, and a target flag
for whether they went on to place a second (repeat) order.

Using only first-order information avoids leaking future purchase data
into the features — this simulates scoring a customer right after their
first purchase, which is the real-world moment a retention/CRM team
would want a propensity score.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DROPPED_STATUSES = {"canceled", "unavailable"}


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "orders": pd.read_csv(RAW_DIR / "olist_orders_dataset.csv", parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]),
        "items": pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv"),
        "products": pd.read_csv(RAW_DIR / "olist_products_dataset.csv"),
        "customers": pd.read_csv(RAW_DIR / "olist_customers_dataset.csv"),
        "category_translation": pd.read_csv(RAW_DIR / "product_category_name_translation.csv"),
        "payments": pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv"),
    }


def build_customer_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"]
    orders = orders[~orders["order_status"].isin(DROPPED_STATUSES)].copy()
    orders = orders.merge(tables["customers"][["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")

    # Target: did this customer place more than one (non-canceled) order?
    order_counts = orders.groupby("customer_unique_id")["order_id"].nunique()
    repeat_customers = set(order_counts[order_counts > 1].index)

    first_orders = (
        orders.sort_values("order_purchase_timestamp")
        .groupby("customer_unique_id", as_index=False)
        .first()
    )
    first_orders["target_repeat_purchase"] = first_orders["customer_unique_id"].isin(repeat_customers).astype(int)

    # Item-level aggregates for the first order
    items = tables["items"].merge(
        tables["products"][["product_id", "product_category_name"]], on="product_id", how="left"
    ).merge(tables["category_translation"], on="product_category_name", how="left")

    item_agg = (
        items.groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            n_distinct_sellers=("seller_id", "nunique"),
            n_distinct_categories=("product_category_name_english", "nunique"),
            items_price=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
        .reset_index()
    )
    # Primary category = category of the highest-priced item in the order
    primary_category = (
        items.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[["order_id", "product_category_name_english"]]
        .rename(columns={"product_category_name_english": "primary_category"})
    )

    # Payment aggregates for the first order
    payment_agg = (
        tables["payments"]
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )

    reviews = tables["reviews"][["order_id", "review_score"]].drop_duplicates("order_id")

    df = first_orders.merge(item_agg, on="order_id", how="left")
    df = df.merge(primary_category, on="order_id", how="left")
    df = df.merge(payment_agg, on="order_id", how="left")
    df = df.merge(reviews, on="order_id", how="left")

    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["late_delivery"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)
    df["purchase_day_of_week"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["freight_ratio"] = df["freight_value"] / df["items_price"].replace(0, pd.NA)

    keep_cols = [
        "customer_unique_id", "customer_state",
        "n_items", "n_distinct_sellers", "n_distinct_categories",
        "items_price", "freight_value", "freight_ratio", "primary_category",
        "payment_value", "payment_installments", "payment_type",
        "review_score", "delivery_days", "late_delivery",
        "purchase_day_of_week", "purchase_month",
        "target_repeat_purchase",
    ]
    return df[keep_cols]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_raw()
    customer_table = build_customer_table(tables)
    out_path = PROCESSED_DIR / "customer_first_order_features.csv"
    customer_table.to_csv(out_path, index=False)

    rate = customer_table["target_repeat_purchase"].mean()
    print(f"Wrote {len(customer_table):,} customers to {out_path}")
    print(f"Repeat-purchase rate: {rate*100:.2f}%")
    print(customer_table.dtypes)


if __name__ == "__main__":
    main()
