import pytest
import pandas as pd


@pytest.fixture(scope="session")
def df():
    return pd.read_csv("data/fraud.csv")
