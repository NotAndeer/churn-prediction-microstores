from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from churn_microstores.config import paths
from churn_microstores.validation import validate_feature_inputs

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Calcula características agregadas por cliente con validaciones básicas."""

    def __init__(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        reference_date: datetime | None = None,
    ):
        self.customers_df = customers_df
        self.events_df = events_df
        self.transactions_df = transactions_df
        self.reference_date = reference_date or self._infer_reference_date()
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        validate_feature_inputs(self.customers_df, self.events_df, self.transactions_df)

    def _infer_reference_date(self) -> datetime:
        if not self.events_df.empty:
            return pd.to_datetime(self.events_df["timestamp"]).max()
        return datetime.now()

    def calculate_customer_features(self) -> pd.DataFrame:
        logger.info("Calculando caracteristicas por cliente")
        features_list: list[dict[str, float | int]] = []

        for customer_id in self.customers_df["customer_id"]:
            customer_events = self.events_df[self.events_df["customer_id"] == customer_id]
            customer_transactions = self.transactions_df[self.transactions_df["customer_id"] == customer_id]

            customer_feats = {
                "customer_id": customer_id,
                "total_sessions": customer_events["session_id"].nunique(),
                "total_events": len(customer_events),
                "avg_session_duration": self._avg_session_duration(customer_events),
                "pages_per_session": self._pages_per_session(customer_events),
                "cart_additions": len(customer_events[customer_events["event_type"] == "add_to_cart"]),
                "product_views": len(customer_events[customer_events["event_type"] == "product_view"]),
                "searches": len(customer_events[customer_events["event_type"] == "search"]),
                "total_purchases": len(customer_transactions),
                "total_spent": customer_transactions["amount"].sum() if len(customer_transactions) > 0 else 0,
                "avg_purchase_value": customer_transactions["amount"].mean() if len(customer_transactions) > 0 else 0,
                "avg_items_per_purchase": customer_transactions["items_count"].mean()
                if len(customer_transactions) > 0
                else 0,
                "days_since_signup": self._days_since_signup(customer_id),
                "days_since_last_activity": self._days_since_last_activity(customer_events),
                "purchase_frequency": self._purchase_frequency(customer_transactions),
            }

            customer_feats["cart_addition_rate"] = customer_feats["cart_additions"] / max(
                1, customer_feats["total_events"]
            )
            customer_feats["purchase_conversion_rate"] = customer_feats["total_purchases"] / max(
                1, customer_feats["total_sessions"]
            )
            customer_feats["avg_days_between_purchases"] = self._avg_days_between_purchases(customer_transactions)

            features_list.append(customer_feats)

        features_df = pd.DataFrame(features_list)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return features_df

    def _avg_session_duration(self, customer_events: pd.DataFrame) -> float:
        if customer_events.empty:
            return 0.0

        session_durations: list[float] = []
        for session_id in customer_events["session_id"].unique():
            session_events = customer_events[customer_events["session_id"] == session_id]
            if len(session_events) > 1:
                duration = (session_events["timestamp"].max() - session_events["timestamp"].min()).total_seconds()
                session_durations.append(float(duration))

        return float(np.mean(session_durations)) if session_durations else 0.0

    def _pages_per_session(self, customer_events: pd.DataFrame) -> float:
        if customer_events.empty:
            return 0.0

        pages_per_session = customer_events.groupby("session_id").size()
        return float(pages_per_session.mean())

    def _days_since_signup(self, customer_id: str) -> int:
        customer = self.customers_df[self.customers_df["customer_id"] == customer_id].iloc[0]
        signup_date = pd.to_datetime(customer["signup_date"])
        return int((self.reference_date - signup_date).days)

    def _days_since_last_activity(self, customer_events: pd.DataFrame) -> int:
        if customer_events.empty:
            return 999

        last_activity = pd.to_datetime(customer_events["timestamp"]).max()
        return int((self.reference_date - last_activity).days)

    def _purchase_frequency(self, customer_transactions: pd.DataFrame) -> float:
        if len(customer_transactions) < 2:
            return 0.0

        timestamps = customer_transactions["timestamp"]
        if isinstance(timestamps.iloc[0], str):
            timestamps = pd.to_datetime(timestamps)

        days_span = (timestamps.max() - timestamps.min()).days
        if days_span == 0:
            return float(len(customer_transactions))

        return float(len(customer_transactions) / (days_span / 30.0))

    def _avg_days_between_purchases(self, customer_transactions: pd.DataFrame) -> float:
        if len(customer_transactions) < 2:
            return 999.0

        sorted_dates = customer_transactions["timestamp"].sort_values()
        if isinstance(sorted_dates.iloc[0], str):
            sorted_dates = pd.to_datetime(sorted_dates)

        days_between = [(sorted_dates.iloc[i] - sorted_dates.iloc[i - 1]).days for i in range(1, len(sorted_dates))]
        return float(np.mean(days_between)) if days_between else 999.0


def create_features(customers_path: Path, events_path: Path, transactions_path: Path, output_path: Path) -> pd.DataFrame:
    logger.info("Cargando datos para feature engineering")

    customers = pd.read_csv(customers_path, parse_dates=["signup_date"])
    events = pd.read_csv(events_path, parse_dates=["timestamp"])
    transactions = pd.read_csv(transactions_path, parse_dates=["timestamp"])

    logger.info("Calculando caracteristicas")
    reference_date = max(events["timestamp"].max(), transactions["timestamp"].max())
    engineer = FeatureEngineer(customers, events, transactions, reference_date=reference_date)
    features_df = engineer.calculate_customer_features()

    logger.info("Guardando caracteristicas")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    logger.info(
        "Caracteristicas creadas: %d caracteristicas para %d clientes", features_df.shape[1], features_df.shape[0]
    )
    return features_df


if __name__ == "__main__":
    from churn_microstores.config import paths
    from churn_microstores.logging_utils import configure_logging

    configure_logging()
    create_features(
        paths.data_raw / "customers.csv",
        paths.data_raw / "events.csv",
        paths.data_raw / "transactions.csv",
        paths.data_processed / "features.csv",
    )
