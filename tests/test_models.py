from churn_microstores.data_generation import DataGenerator
from churn_microstores.features import create_features
from churn_microstores.models import ChurnPredictor


def setup_dataset(tmp_path, n_customers: int = 80):
    generator = DataGenerator(n_customers=n_customers, random_state=7)
    generator.generate_all_data()
    create_features(
        tmp_path / "customers.csv",
        tmp_path / "events.csv",
        tmp_path / "transactions.csv",
        tmp_path / "features.csv",
    )


def test_model_pipeline_trains(monkeypatch, tmp_path):
    monkeypatch.setattr("churn_microstores.config.paths.data_raw", tmp_path)
    monkeypatch.setattr("churn_microstores.config.paths.data_processed", tmp_path)
    setup_dataset(tmp_path)

    predictor = ChurnPredictor()
    X_train, X_test, y_train, y_test, _ = predictor.prepare_data(
        tmp_path / "features.csv", tmp_path / "churn_labels.csv"
    )
    evaluation = predictor.train_models(X_train, X_test, y_train, y_test)

    assert not evaluation.empty


def test_model_auc_above_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr("churn_microstores.config.paths.data_raw", tmp_path)
    monkeypatch.setattr("churn_microstores.config.paths.data_processed", tmp_path)
    setup_dataset(tmp_path, n_customers=100)

    predictor = ChurnPredictor()
    X_train, X_test, y_train, y_test, _ = predictor.prepare_data(
        tmp_path / "features.csv", tmp_path / "churn_labels.csv"
    )
    evaluation = predictor.train_models(X_train, X_test, y_train, y_test)

    assert evaluation["AUC"].max() > 0.5
