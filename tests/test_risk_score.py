import unittest
import numpy as np
import pandas as pd

from src.scoring import compute_risk_score
from src.scoring import compute_percentage_fraud


class TestRiskEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the transaction dataset once for all tests in this class
        # This improves efficiency by avoiding repeated file I/O operations
        cls.df = pd.read_csv("data/fraud.csv")

    def test_score_bounds(self):
        # Test that all computed fraud risk scores are properly normalised
        # Risk scores should always fall within [0, 1] for interpretability
        # 0 = low risk, 1 = high risk

        scores = compute_risk_score(self.df)

        # Ensure no score is below 0
        self.assertTrue((scores >= 0).all())

        # Ensure no score is above 1
        self.assertTrue((scores <= 1).all())

    def test_score_stability(self):
        # Test that the risk scoring function is deterministic
        # Running the same input multiple times should produce consistent outputs
        # This is critical for auditability in fraud detection systems

        scores1 = compute_risk_score(self.df)
        scores2 = compute_risk_score(self.df)

        # Measure average absolute difference between runs
        diff = np.mean(np.abs(scores1 - scores2))

        # Allow a very small tolerance for floating-point precision issues
        self.assertLess(
            diff,
            0.05,
            "Risk score is unstable — results vary too much between runs"
        )

    def test_score_balance_0(self):
        scores = compute_percentage_fraud(self.df)
        
        self.assertNotEqual(scores, 1, 'Impossible for this to occur!')

    
    def test_score_balance_100(self):
        scores = compute_percentage_fraud(self.df)
        
        self.assertNotEqual(scores, 0, 'Impossible for this to occur!')


if __name__ == "__main__":
    unittest.main()