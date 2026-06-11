"""
Stage 1 — Statistical Investigation.

Forensic statistics that surface suspicious signal without using fraud labels:
  - Univariate distribution summaries and skewness/kurtosis flags
  - Benford's Law test on transaction_amount (first-digit distribution)
  - Bivariate correlation heatmap
  - Missing-value pattern analysis (MAR vs MNAR suspicion)
  - Distribution comparison: label-split box plots (for post-hoc interpretation only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional


# ── Univariate summary ────────────────────────────────────────────────────────

def univariate_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a per-feature forensic summary table."""
    rows = []
    for col in features:
        s = df[col].dropna()
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        missing_pct = df[col].isna().mean() * 100
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        # flag potential anomaly: extreme skew or high kurtosis
        flag = abs(skew) > 2 or abs(kurt) > 7 or missing_pct > 10
        rows.append({
            "feature": col,
            "n_valid": len(s),
            "missing_%": round(missing_pct, 1),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "p25": round(float(s.quantile(0.25)), 3),
            "p50": round(float(s.median()), 3),
            "p75": round(float(s.quantile(0.75)), 3),
            "max": round(float(s.max()), 3),
            "iqr": round(iqr, 3),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "forensic_flag": flag,
        })
    return pd.DataFrame(rows).set_index("feature")


# ── Benford's Law ─────────────────────────────────────────────────────────────

def benfords_law_test(
    series: pd.Series,
    label: str = "transaction_amount",
    ax: Optional[plt.Axes] = None,
) -> dict:
    """
    Chi-square goodness-of-fit test against Benford's expected first-digit distribution.

    Returns dict with chi2 statistic, p-value, and per-digit observed/expected counts.
    Low p-value (~<0.05) suggests the distribution has been manipulated.
    """
    s = series.dropna()
    # extract first significant digit
    leading = s[s > 0].apply(lambda x: int(str(x).lstrip("0").replace(".", "")[0]))
    observed_counts = leading.value_counts().sort_index().reindex(range(1, 10), fill_value=0)

    benford_probs = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
    expected_counts = benford_probs * len(leading)

    chi2, p_value = stats.chisquare(observed_counts.values, f_exp=expected_counts)

    if ax is not None:
        digits = range(1, 10)
        x = np.arange(len(digits))
        width = 0.35
        ax.bar(x - width / 2, observed_counts.values / len(leading),
               width, label="Observed", color="#4C72B0", alpha=0.85)
        ax.bar(x + width / 2, benford_probs,
               width, label="Benford expected", color="#DD8452", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(digits)
        ax.set_xlabel("First digit")
        ax.set_ylabel("Proportion")
        ax.set_title(f"Benford's Law — {label}\n"
                     f"χ²={chi2:.2f}, p={p_value:.4f}"
                     f"{'  ⚠ SUSPICIOUS' if p_value < 0.05 else ''}")
        ax.legend()

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "suspicious": p_value < 0.05,
        "observed_proportions": (observed_counts / len(leading)).round(4).to_dict(),
        "benford_proportions": dict(zip(range(1, 10), benford_probs.round(4))),
    }


# ── Missing-value pattern ─────────────────────────────────────────────────────

def missing_value_heatmap(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> pd.DataFrame:
    """
    Plot a missing-value co-occurrence matrix.
    Correlated missingness across features hints at MNAR — a forensic signal.
    """
    miss = df.isnull().astype(int)
    corr = miss.corr()

    if ax is not None:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, ax=ax, cmap="Reds", vmin=0, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5,
        )
        ax.set_title("Missing-value correlation\n(high values → data may be MNAR)")

    return corr


# ── Bivariate correlation ─────────────────────────────────────────────────────

def correlation_heatmap(
    df: pd.DataFrame, features: list[str], ax: Optional[plt.Axes] = None
) -> pd.DataFrame:
    corr = df[features].corr()
    if ax is not None:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, vmin=-1, vmax=1,
        )
        ax.set_title("Feature correlation matrix")
    return corr


# ── Label-split distributions (post-hoc only) ────────────────────────────────

def label_split_boxplots(
    df: pd.DataFrame, features: list[str], label_col: str = "is_fraud"
) -> plt.Figure:
    """
    Box plots split by fraud label — used only for post-hoc interpretation,
    not to inform the unsupervised pipeline.
    """
    n = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        data = [
            df.loc[df[label_col] == 0, col].dropna().values,
            df.loc[df[label_col] == 1, col].dropna().values,
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_title(col, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions by fraud label (post-hoc reference)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ── Normality tests ───────────────────────────────────────────────────────────

def normality_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Shapiro-Wilk (n≤5000) or D'Agostino-Pearson on each feature."""
    rows = []
    for col in features:
        s = df[col].dropna()
        n = len(s)
        sample = s.sample(min(n, 5000), random_state=42)
        stat, p = stats.normaltest(sample)
        rows.append({
            "feature": col,
            "test": "D'Agostino-Pearson",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "normal": p > 0.05,
        })
    return pd.DataFrame(rows).set_index("feature")
