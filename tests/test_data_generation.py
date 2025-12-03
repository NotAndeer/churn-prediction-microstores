import pandas as pd

from churn_microstores.data_generation import DataGenerator


def test_generate_exact_customer_count(tmp_path, monkeypatch):
    monkeypatch.setattr("churn_microstores.config.paths.data_raw", tmp_path)
    generator = DataGenerator(n_customers=100, random_state=123)
    generated = generator.generate_all_data()
    assert len(generated.customers) == 100


def test_generated_tables_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("churn_microstores.config.paths.data_raw", tmp_path)
    generator = DataGenerator(n_customers=50, random_state=123)
    generated = generator.generate_all_data()

    for df in [generated.customers, generated.events, generated.transactions, generated.churn_labels]:
        assert "customer_id" in df.columns

    assert generated.churn_labels["churn"].dtype in [int, "int64", "int32"]
