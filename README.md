# Fraud Detection Pipeline

An unsupervised forensic fraud detection pipeline that ranks transactions by risk without using fraud labels during modelling.

## Overview

The pipeline has three stages:

1. **Statistical Investigation** — forensic statistics on the raw data: missing value patterns, Benford's Law test on transaction amounts, normality tests, and feature correlations. Surfaces suspicious signal before any model is fitted.

2. **Anomaly Detection** — an Isolation Forest fitted on the full imputed feature matrix. No fraud labels used. Transactions that are easy to isolate from the rest are scored as anomalous.

3. **Risk Scoring** — raw anomaly scores are inverted and normalised to [0, 1]. Transactions are bucketed into three tiers calibrated to the ~10% fraud base rate:

| Tier | Label | Threshold |
|------|-------|-----------|
| Low | 0 | Bottom 60% |
| Medium | 1 | 60th–90th percentile |
| High | 2 | Top 10% |

## Project Structure

```
├── data/
│   └── fraud.csv               # 7,000 transactions, 12 features
├── src/
│   ├── fraud/
│   │   ├── data_loader.py      # load, impute, feature_matrix
│   │   └── statistical_investigation.py  # Stage 1 analysis functions
│   ├── scoring.py              # compute_risk_score, create_pseudo_labels, compute_percentage_fraud
│   └── pipeline.py             # run_full_pipeline
├── tests/
│   ├── conftest.py             # session-scoped df fixture
│   ├── test_risk_score.py
│   ├── test_psuedo_labels.py
│   ├── test_end_to_end.py
│   ├── test_model_anomalies.py
│   └── test_statistical_signals.py
├── experiments.ipynb           # full narrative pipeline notebook
└── .scratch/
    └── prd-fraud-detection-pipeline.md
```

## Quickstart

```python
from src.fraud.data_loader import load
from src.pipeline import run_full_pipeline
from src.scoring import create_pseudo_labels, compute_percentage_fraud

df = load()
results = run_full_pipeline(df)      # original df + risk_score column
labels = create_pseudo_labels(df)    # 0 / 1 / 2 risk tier per row
pct = compute_percentage_fraud(df)   # estimated % high-risk transactions

# Top 10 highest-risk transactions
results.sort_values("risk_score", ascending=False).head(10)
```

## Running the Notebook

Open `experiments.ipynb` in Jupyter. The notebook runs all three stages end-to-end and concludes with post-hoc evaluation against the held-out fraud label.

## Running Tests

```bash
python -m pytest tests/ -v
```

All 16 tests should pass.

## Post-hoc Performance

Evaluated against the held-out `is_fraud` label after the unsupervised pipeline was built:

- **AUC-ROC**: 0.516
- **Estimated high-risk rate**: 10.0%

The AUC reflects the difficulty of the unsupervised setting — no labels were used to tune the model. The tier calibration correctly targets the ~10% fraud base rate.

## Dataset

`data/fraud.csv` — 7,000 transactions with 12 behavioural features:

| Feature | Description |
|---|---|
| `transaction_amount` | Value of the transaction |
| `hour_of_day` | Hour the transaction occurred |
| `is_weekend` | Whether the transaction was on a weekend |
| `num_items` | Number of items purchased |
| `customer_age` | Age of the customer |
| `prev_transactions` | Number of prior transactions by this customer |
| `distance_from_home` | Distance of transaction from customer's home |
| `device_type` | Device used (0 = desktop, 1 = mobile, 2 = other) |
| `network_quality` | Network signal quality score |
| `is_first_transaction` | Whether this is the customer's first transaction |
| `store_type` | Type of store (0 = physical, 1 = online) |
| `velocity_score` | Transaction velocity signal |
