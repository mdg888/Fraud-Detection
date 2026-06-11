# PRD: Unsupervised Fraud Detection Pipeline

**Label:** `ready-for-agent`

---

## Problem Statement

A fraud analyst has a dataset of 7,000 financial transactions with rich behavioural features (transaction amount, velocity score, distance from home, device type, etc.) but **no reliable fraud labels** to train a supervised classifier. The existing codebase has data loading and statistical investigation utilities, but no scoring, pseudo-labelling, or end-to-end pipeline — meaning there is currently no way to rank transactions by fraud risk or surface the highest-risk cases for review.

The analyst needs a system that can ingest raw transaction data, detect anomalous behaviour without relying on labels, assign each transaction a normalised risk score, and bucket transactions into actionable risk tiers — all in a way that is deterministic, auditable, and testable.

---

## Solution

Build three public modules that together form a complete unsupervised fraud detection pipeline:

1. A **risk scoring module** (`src/scoring.py`) that fits an Isolation Forest on the transaction feature matrix and produces a normalised [0, 1] anomaly-based risk score per transaction.
2. A **pseudo-labelling function** within that module that buckets risk scores into three tiers (low / medium / high) calibrated to the dataset's ~10% fraud base rate.
3. A **pipeline module** (`src/pipeline.py`) that wires data loading, imputation, scoring, and pseudo-labelling into a single callable that accepts raw data and returns it annotated with a risk score.

Wire these into a `conftest.py`-based test suite that verifies all public behaviors against the real dataset.

---

## User Stories

1. As a fraud analyst, I want each transaction assigned a risk score between 0 and 1, so that I can rank transactions from least to most suspicious without needing labelled training data.
2. As a fraud analyst, I want the risk score to be deterministic, so that I can re-run the pipeline at any time and get the same results for audit purposes.
3. As a fraud analyst, I want transactions bucketed into low (0), medium (1), and high (2) risk tiers, so that I can triage my review queue into actionable priority groups.
4. As a fraud analyst, I want the high-risk tier to capture approximately the top 10% of transactions, so that the tier size is calibrated to the known base rate rather than an arbitrary equal split.
5. As a fraud analyst, I want a single function that tells me the estimated percentage of fraudulent transactions in a dataset, so that I can quickly assess the overall risk level of a new batch.
6. As a fraud analyst, I want the estimated fraud percentage to never be 0% or 100%, so that I can trust the system is not producing degenerate outputs on real-world data.
7. As a fraud analyst, I want to run the full pipeline on a raw DataFrame in one call, so that I do not need to manually coordinate imputation, feature extraction, and scoring steps.
8. As a fraud analyst, I want the pipeline to return the original transaction data with the risk score appended, so that I can join the scores back to the raw records without data loss.
9. As a fraud analyst, I want missing feature values handled automatically inside the pipeline, so that I do not need to pre-process data before passing it to the scoring function.
10. As a fraud analyst, I want the top 10 highest-risk transactions to be easily retrievable by sorting on the risk score column, so that I can immediately focus review effort on the most suspicious cases.
11. As a data scientist, I want the risk scoring logic isolated in its own module with a clean public interface, so that I can swap the underlying anomaly model later without breaking callers.
12. As a data scientist, I want pseudo-labels that reflect the dataset's base rate, so that downstream supervised models trained on pseudo-labels start from a realistic class distribution.
13. As a data scientist, I want the pipeline to leave the original DataFrame unmodified (imputation is internal), so that callers are not surprised by silently mutated input data.
14. As a developer, I want a session-scoped pytest fixture that loads the real dataset once, so that the test suite runs efficiently without redundant file I/O.
15. As a developer, I want all public interfaces covered by behavior-level tests, so that I can refactor internal model choices without rewriting tests.

---

## Implementation Decisions

- **`src/scoring.py`** is a new top-level module (not inside `src/fraud/`). It imports `impute` and `feature_matrix` from `src/fraud/data_loader` rather than duplicating that logic.

- **`compute_risk_score(df) -> pd.Series`**: fits `IsolationForest(contamination=0.103, random_state=42)` on the imputed feature matrix, inverts and normalises `decision_function` output to [0, 1]. Deterministic by construction via `random_state`.

- **`create_pseudo_labels(df) -> pd.Series`**: calls `compute_risk_score` internally, then applies fixed percentile thresholds — bottom 60% → 0 (low), next 30% → 1 (medium), top 10% → 2 (high). Thresholds are anchored to the dataset's ~10.3% fraud base rate.

- **`compute_percentage_fraud(df) -> float`**: calls `create_pseudo_labels` internally and returns `(label == 2).mean() * 100`. On any realistic transaction dataset this will be strictly between 0 and 100.

- **`src/pipeline.py`** exposes a single function **`run_full_pipeline(df) -> pd.DataFrame`**. Internally: impute → feature_matrix → score → attach `risk_score` column to the *original* (pre-imputation) df. The returned DataFrame is a copy; the input is never mutated.

- **`tests/conftest.py`** provides a `df` session-scoped fixture that reads `data/fraud.csv` once per pytest session and is shared across all test classes/functions.

- Existing test files (`test_statistical_signals.py`, `test_psuedo_labels.py`, `test_end_to_end.py`, `test_risk_score.py`) are updated to: use `data/fraud.csv` (not `data/transactions.csv`), use `assertNotEqual` (not deprecated `assertNotEquals`), and import from the correct module paths.

- `test_model_anomalies.py` is left as-is per the developer's instruction (column bug already fixed by the developer).

---

## Testing Decisions

**What makes a good test here:** tests verify behavior through the public interface (`compute_risk_score`, `create_pseudo_labels`, `compute_percentage_fraud`, `run_full_pipeline`). They do not assert on internal model parameters, intermediate imputed values, or sklearn internals. A test should survive a full internal refactor (e.g. swapping Isolation Forest for LOF) as long as the public contract holds.

**Modules under test and their key behaviors:**

- `compute_risk_score`: output length equals input length; all scores in [0, 1]; two calls on the same input produce identical output (determinism).
- `create_pseudo_labels`: output length equals input length; all values in {0, 1, 2}; value counts sum to total row count (no nulls, no dropped rows).
- `compute_percentage_fraud`: result is strictly greater than 0 and strictly less than 100 on the real dataset.
- `run_full_pipeline`: output row count equals input row count; output contains a `risk_score` column; all `risk_score` values in [0, 1]; top 10 rows by `risk_score` is retrievable (len == 10).

**Test fixture:** `conftest.py` session-scoped `df` fixture loads `data/fraud.csv` once. All test classes consume it rather than calling `pd.read_csv` per class.

---

## Out of Scope

- Supervised model training using the `is_fraud` label — the label exists in the data but must not be used to train or tune the pipeline.
- Post-hoc evaluation metrics (AUC-ROC, precision/recall against `is_fraud`) — this belongs in a separate evaluation module.
- The `src/fraud/statistical_investigation.py` module — already implemented, not part of this PRD.
- Hyperparameter tuning of the Isolation Forest — `contamination=0.103` and `random_state=42` are fixed for this iteration.
- A Jupyter notebook narrative — the notebook (`experiments.ipynb`) is a separate deliverable.
- Deployment, serving, or batch scoring infrastructure.

---

## Further Notes

- The dataset has significant missingness (up to 15% on `velocity_score`). Median imputation inside the pipeline is a pragmatic choice for now; a more principled approach (MICE, iterative imputer) is a natural follow-up.
- The `velocity_score` and `distance_from_home` features are strong domain signals. A future iteration should consider blending these explicitly into the risk score rather than leaving them as implicit Isolation Forest inputs.
- The pseudo-label thresholds (60/30/10) are hardcoded to this dataset's base rate. If the pipeline is applied to a different dataset with a different fraud rate, these thresholds should be made configurable.
