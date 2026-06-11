import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class TestAnomalyModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load dataset once
        # This simulates real transaction data used for fraud detection
        cls.df = pd.read_csv("data/fraud.csv")

    def test_isolation_forest_executes(self):
        # ---------------------------------------------
        # PURPOSE:
        # Ensure Isolation Forest model can be trained without errors
        # ---------------------------------------------

        # Select only numerical features for anomaly detection
        # (simplified fraud signal using transaction transaction_transaction_amount)
        X = self.df[["transaction_amount"]].fillna(0)

        # Initialise Isolation Forest
        # contamination = expected proportion of fraud/anomalies
        # random_state ensures reproducibility for audits
        model = IsolationForest(contamination=0.05, random_state=42)

        # Train model on transaction data (unsupervised learning)
        model.fit(X)

        # Generate anomaly scores (lower = more anomalous)
        scores = model.decision_function(X)

        # Ensure model produces output for every transaction
        self.assertEqual(len(scores), len(self.df))

    def test_anomaly_scores_are_finite(self):
        # ---------------------------------------------
        # PURPOSE:
        # Ensure model outputs valid numeric scores
        # ---------------------------------------------

        X = self.df[["transaction_amount"]].fillna(0)

        model = IsolationForest(random_state=42)
        model.fit(X)

        scores = model.decision_function(X)

        # Ensure no NaN or infinite values exist
        self.assertTrue(
            np.isfinite(scores).all(),
            "Anomaly scores contain invalid values (NaN or inf)"
        )

    def test_anomaly_score_variance_exists(self):
        # ---------------------------------------------
        # PURPOSE:
        # Ensure the model actually distinguishes between transactions
        # If all scores are similar → model is useless for fraud detection
        # ---------------------------------------------

        X = self.df[["transaction_amount"]].fillna(0)

        model = IsolationForest(random_state=42)
        model.fit(X)

        scores = model.decision_function(X)

        # Standard deviation checks if there is meaningful variation
        self.assertGreater(
            np.std(scores),
            0.001,
            "Model does not differentiate between transactions"
        )


if __name__ == "__main__":
    unittest.main()