import unittest
import pandas as pd

from src.scoring import create_pseudo_labels


class TestPseudoLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load transaction dataset once for all test cases
        # This avoids reloading data for every individual test (improves efficiency)
        cls.df = pd.read_csv("data/fraud.csv")

    def test_label_length(self):
        # Test that pseudo-label generation produces exactly one label per transaction
        # This ensures no rows are dropped or duplicated during label creation

        labels = create_pseudo_labels(self.df)

        self.assertEqual(
            len(labels),
            len(self.df),
            "Mismatch: number of labels must equal number of transactions"
        )

    def test_label_values(self):
        # Test that pseudo-labels only contain valid classes
        # Expected encoding:
        # 0 = Low risk
        # 1 = Medium risk
        # 2 = High risk

        labels = create_pseudo_labels(self.df)

        self.assertTrue(
            set(labels.unique()).issubset({0, 1, 2}),
            "Invalid label detected: labels must be in {0, 1, 2}"
        )

    def test_label_distribution(self):
        # Test that all generated labels account for every transaction
        # Ensures no null labels or dropped records exist

        labels = create_pseudo_labels(self.df)

        self.assertEqual(
            labels.value_counts().sum(),
            len(self.df),
            "Label distribution mismatch: some transactions may be unlabelled"
        )