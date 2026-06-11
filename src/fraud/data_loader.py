from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parents[2] / "data" / "fraud.csv"

NUMERIC_FEATURES = [
    "transaction_amount",
    "hour_of_day",
    "is_weekend",
    "num_items",
    "customer_age",
    "prev_transactions",
    "distance_from_home",
    "device_type",
    "network_quality",
    "is_first_transaction",
    "store_type",
    "velocity_score",
]

LABEL_COL = "is_fraud"


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute all numeric features; does not touch the label column."""
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) — X has no label column, y is the held-out fraud label."""
    X = df[NUMERIC_FEATURES].copy()
    y = df[LABEL_COL].copy() if LABEL_COL in df.columns else pd.Series(dtype=int)
    return X, y
