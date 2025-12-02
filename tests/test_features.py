import pandas as pd

from churn_microstores.data_generation import DataGenerator
from churn_microstores.features import create_features


def test_create_features_unique_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("churn_microstores.config.paths.data_raw", tmp_path)
    monkeypatch.setattr("churn_microstores.config.paths.data_processed", tmp_path)
    generator = DataGenerator(n_customers=30, random_state=42)
    generated = generator.generate_all_data()

    features_df = create_features(
        tmp_path / "customers.csv",
        tmp_path / "events.csv",
        tmp_path / "transactions.csv",
        tmp_path / "features.csv",
    )

    assert features_df["customer_id"].is_unique


def test_create_features_no_nans(tmp_path, monkeypatch):
    monkeypatch.setattr("churn_microstores.config.paths.data_raw", tmp_path)
    monkeypatch.setattr("churn_microstores.config.paths.data_processed", tmp_path)
    generator = DataGenerator(n_customers=20, random_state=99)
    generator.generate_all_data()

    features_df = create_features(
        tmp_path / "customers.csv",
        tmp_path / "events.csv",
        tmp_path / "transactions.csv",
        tmp_path / "features.csv",
    )

    key_cols = ["total_spent", "total_sessions", "days_since_last_activity"]
    assert not features_df[key_cols].isnull().any().any()
