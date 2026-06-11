from __future__ import annotations

import pandas as pd

from src.scoring import compute_risk_score


def run_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a normalised risk_score column appended."""
    out = df.copy()
    out["risk_score"] = compute_risk_score(df)
    return out
