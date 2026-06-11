import unittest
import pandas as pd

# This function represents your full fraud detection pipeline:
# preprocessing → feature engineering → anomaly detection → risk scoring
from src.pipeline import run_full_pipeline


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load raw transaction data once for all tests
        # This simulates real-world input as it would arrive in production
        cls.df = pd.read_csv("data/fraud.csv")

    def test_pipeline_executes_without_errors(self):
        # ---------------------------------------------
        # PURPOSE:
        # Ensure the entire fraud detection pipeline runs end-to-end
        # without crashing or breaking at any stage
        # ---------------------------------------------

        output = run_full_pipeline(self.df)

        # The pipeline must return a dataframe-like object
        # with the same number of rows as input
        self.assertEqual(len(output), len(self.df))

        # The pipeline MUST produce a fraud risk score column
        self.assertIn(
            "risk_score",
            output.columns,
            "Pipeline must output a risk_score column"
        )

    def test_pipeline_output_validity(self):
        # ---------------------------------------------
        # PURPOSE:
        # Ensure the final output contains valid fraud scores
        # ---------------------------------------------

        output = run_full_pipeline(self.df)

        # Risk scores should always be normalised between 0 and 1
        self.assertTrue((output["risk_score"] >= 0).all())
        self.assertTrue((output["risk_score"] <= 1).all())

    def test_top_anomalies_exist(self):
        # ---------------------------------------------
        # PURPOSE:
        # Ensure the system can rank and identify high-risk transactions
        # ---------------------------------------------

        output = run_full_pipeline(self.df)

        # Sort transactions by highest fraud risk
        top10 = output.sort_values("risk_score", ascending=False).head(10)

        # We expect a top-10 list to always exist
        self.assertEqual(len(top10), 10)


if __name__ == "__main__":
    unittest.main()