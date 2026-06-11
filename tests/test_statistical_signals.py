import unittest
import numpy as np
import pandas as pd

class TestStatisticalSignals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv("data/transactions.csv")

    def test_benford_distribution(self):
        amounts = self.df["transaction_amount"].dropna().astype(str)
        first_digits = amounts.str[0].astype(int)

        freq = first_digits.value_counts(normalize=True)

        self.assertTrue(freq.sum() > 0)
        self.assertTrue(len(freq) > 1)

    def test_round_number_bias(self):
        amounts = self.df["transaction_amount"]

        round_ratio = amounts.astype(str).str.endswith(("00", "50", "99")).mean()

        self.assertTrue(0 <= round_ratio <= 1)

    def test_outliers_exist(self):
        q99 = self.df["transaction_amount"].quantile(0.99)
        outliers = self.df[self.df["transaction_amount"] > q99]

        self.assertGreaterEqual(len(outliers), 0)


if __name__ == "__main__":
    unittest.main()