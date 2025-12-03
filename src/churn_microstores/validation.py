from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_CUSTOMER_COLS = {"customer_id", "signup_date"}
REQUIRED_EVENT_COLS = {"customer_id", "session_id", "timestamp", "event_type"}
REQUIRED_TX_COLS = {"customer_id", "timestamp", "amount", "items_count"}
REQUIRED_FEATURE_COLS = {
    "customer_id",
    "total_spent",
    "total_events",
    "total_sessions",
    "days_since_last_activity",
}


def _check_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {df_name}: {missing}")


def validate_feature_inputs(customers_df: pd.DataFrame, events_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
    _check_columns(customers_df, REQUIRED_CUSTOMER_COLS, "clientes")
    _check_columns(events_df, REQUIRED_EVENT_COLS, "eventos")
    _check_columns(transactions_df, REQUIRED_TX_COLS, "transacciones")


def validate_features_schema(features_df: pd.DataFrame) -> None:
    _check_columns(features_df, REQUIRED_FEATURE_COLS, "caracteristicas")

    if (features_df["days_since_last_activity"] < 0).any():
        raise ValueError("days_since_last_activity no puede ser negativo")
    if (features_df["total_spent"] < 0).any():
        raise ValueError("total_spent no puede ser negativo")


def validate_label_ranges(labels_df: pd.DataFrame) -> None:
    if "churn" in labels_df.columns and not set(labels_df["churn"].unique()).issubset({0, 1}):
        raise ValueError("churn debe ser 0 o 1")


__all__ = [
    "validate_feature_inputs",
    "validate_features_schema",
    "validate_label_ranges",
    "REQUIRED_CUSTOMER_COLS",
    "REQUIRED_EVENT_COLS",
    "REQUIRED_TX_COLS",
    "REQUIRED_FEATURE_COLS",
]
