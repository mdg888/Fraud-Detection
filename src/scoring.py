from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.fraud.data_loader import impute, feature_matrix


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Return a normalised [0, 1] anomaly-based risk score for each row."""
    X, _ = feature_matrix(impute(df))
    model = IsolationForest(contamination=0.103, random_state=42)
    model.fit(X)
    raw = model.decision_function(X)
    # decision_function: lower = more anomalous; invert and normalise to [0, 1]
    inverted = -raw
    min_val, max_val = inverted.min(), inverted.max()
    normalised = (inverted - min_val) / (max_val - min_val)
    return pd.Series(normalised, index=df.index)


def create_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    """Bucket risk scores into 0 (low), 1 (medium), 2 (high) risk tiers.

    Thresholds calibrated to ~10% fraud base rate:
    bottom 60% → 0, next 30% → 1, top 10% → 2.
    """
    scores = compute_risk_score(df)
    p60 = scores.quantile(0.60)
    p90 = scores.quantile(0.90)
    labels = pd.Series(0, index=df.index)
    labels[scores >= p60] = 1
    labels[scores >= p90] = 2
    return labels


def compute_percentage_fraud(df: pd.DataFrame) -> float:
    """Return the estimated percentage of high-risk (tier 2) transactions."""
    labels = create_pseudo_labels(df)
    return float((labels == 2).mean() * 100)
