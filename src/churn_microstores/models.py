from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn_microstores.config import paths
from churn_microstores.io_utils import load_features, load_labels
from churn_microstores.metrics_utils import save_metrics

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Administra el ciclo completo de entrenamiento y evaluación de modelos."""

    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        self.feature_names: list[str] = []
        self.best_model_name: str | None = None
        self.evaluation_results: pd.DataFrame | None = None

    def prepare_data(self, features_path: Path, labels_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        logger.info("Preparando datos para modelado")
        features_df = load_features(features_path)
        labels_df = load_labels(labels_path)

        data = features_df.merge(labels_df[["customer_id", "churn"]], on="customer_id")
        X = data.drop(["customer_id", "churn"], axis=1)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna()
        y = data.loc[X.index, "churn"]

        self.feature_names = X.columns.tolist()
        logger.debug("Características después de limpieza: %s", self.feature_names)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test, data

    def _train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        return model

    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        return model

    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(
            n_estimators=160,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train.astype(np.float32), y_train)
        return model

    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        logger.info("Entrenando modelos")

        self.models["logistic_regression"] = self._train_logistic_regression(X_train, y_train)
        self.models["random_forest"] = self._train_random_forest(X_train, y_train)
        self.models["xgboost"] = self._train_xgboost(X_train, y_train)

        evaluation = self.evaluate_models(X_test, y_test)
        self.best_model_name = evaluation.loc[evaluation["AUC"].idxmax(), "Modelo"]
        self.evaluation_results = evaluation
        return evaluation

    def _compute_metrics(self, model_name: str, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        X_eval = X_test.astype(np.float32) if model_name == "xgboost" else X_test
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1]
        return {
            "auc": float(roc_auc_score(y_test, y_pred_proba)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        logger.info("Evaluando modelos")
        results = []
        for name, model in self.models.items():
            metrics = self._compute_metrics(name, model, X_test, y_test)
            results.append({"Modelo": name, "AUC": metrics["auc"], "Accuracy": metrics["accuracy"]})
            logger.info("%s -> AUC: %.3f | Accuracy: %.3f", name, metrics["auc"], metrics["accuracy"])
            if name == "xgboost":
                X_eval = X_test.astype(np.float32)
                y_pred = model.predict(X_eval)
                logger.debug("Reporte de clasificación - %s:\n%s", name, classification_report(y_test, y_pred))

            save_metrics(metrics, paths.data_processed, name)

        results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
        return results_df

    def save_evaluation_report(self, results_df: pd.DataFrame, output_path: Path | None = None) -> None:
        output_path = output_path or (paths.data_processed / "model_report.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info("Reporte de evaluación guardado en %s", output_path)

    def predict_churn(self, customer_data: pd.Series | np.ndarray) -> Tuple[int, float]:
        if not self.models:
            raise ValueError("Primero entrena los modelos")

        model_name = self.best_model_name or "xgboost"
        model = self.models.get(model_name)
        if model is None:
            raise ValueError("No se encontró un modelo entrenado para predecir")

        if hasattr(customer_data, "values"):
            customer_df = customer_data.to_frame().T
        else:
            customer_df = pd.DataFrame([customer_data], columns=self.feature_names)

        customer_df = customer_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        prediction = int(model.predict(customer_df)[0])
        probability = float(model.predict_proba(customer_df)[:, 1][0])
        return prediction, probability


def run_model_pipeline() -> Tuple[ChurnPredictor, Dict[str, object]]:
    logger.info("Iniciando pipeline de modelado")
    predictor = ChurnPredictor()

    X_train, X_test, y_train, y_test, _ = predictor.prepare_data(
        paths.data_processed / "features.csv", paths.data_raw / "churn_labels.csv"
    )

    logger.info("Datos para entrenamiento: %d clientes", X_train.shape[0])
    logger.info("Datos para prueba: %d clientes", X_test.shape[0])
    logger.info("Características: %d", X_train.shape[1])
    logger.info("Tasa de churn: %.1f%%", y_train.mean() * 100)

    evaluation = predictor.train_models(X_train, X_test, y_train, y_test)
    predictor.save_evaluation_report(evaluation)

    logger.info("Pipeline de modelado completado")
    return predictor, predictor.models


if __name__ == "__main__":
    from churn_microstores.logging_utils import configure_logging

    configure_logging()
    run_model_pipeline()
