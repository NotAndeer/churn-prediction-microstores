"""Synthetic data generation utilities for churn microstores."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from churn_microstores.config import generation, paths

logger = logging.getLogger(__name__)


@dataclass
class GeneratedData:
    customers: pd.DataFrame
    events: pd.DataFrame
    transactions: pd.DataFrame
    churn_labels: pd.DataFrame


class DataGenerator:
    """Genera un dataset sintético coherente para pruebas de churn."""

    def __init__(self, n_customers: int = generation.n_customers, random_state: int | None = generation.random_seed):
        self.n_customers = n_customers
        self.random_state = random_state
        self.customers_df: pd.DataFrame | None = None
        self.events_df: pd.DataFrame | None = None
        self.transactions_df: pd.DataFrame | None = None
        self._set_random_seed()
        self._ensure_output_dirs()

    def _set_random_seed(self) -> None:
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            logger.debug("Random seed set to %s", self.random_state)

    def _ensure_output_dirs(self) -> None:
        paths.data_raw.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured raw data directory exists at %s", paths.data_raw)

    def generate_customer_profiles(self) -> pd.DataFrame:
        logger.info("Generando perfiles de clientes para %d clientes", self.n_customers)
        profiles = []
        today = datetime.now()
        locations = ["Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena"]
        devices = ["mobile", "desktop", "tablet"]
        age_groups = ["18-25", "26-35", "36-45", "46+"]

        for i in range(self.n_customers):
            signup_date = today - timedelta(days=random.randint(1, 365))
            profiles.append(
                {
                    "customer_id": f"CUST_{i:06d}",
                    "signup_date": signup_date,
                    "location": random.choice(locations),
                    "device_type": random.choice(devices),
                    "age_group": random.choice(age_groups),
                }
            )

        self.customers_df = pd.DataFrame(profiles)
        return self.customers_df

    def generate_behavioral_data(self, days: int = 180) -> pd.DataFrame:
        logger.info("Generando datos de comportamiento")
        if self.customers_df is None:
            raise ValueError("Primero genera los perfiles de clientes")

        events = []
        event_types = ["page_view", "add_to_cart", "product_view", "search", "wishlist_add"]

        for _, customer in self.customers_df.iterrows():
            n_sessions = max(1, int(np.random.poisson(8)))

            for session in range(n_sessions):
                session_id = f"SESS_{customer['customer_id']}_{session}"
                session_date = customer["signup_date"] + timedelta(
                    days=random.randint(0, (datetime.now() - customer["signup_date"]).days)
                )

                n_events = random.randint(3, 20)
                for event_num in range(n_events):
                    event_time = session_date + timedelta(minutes=event_num * 2)

                    events.append(
                        {
                            "customer_id": customer["customer_id"],
                            "session_id": session_id,
                            "timestamp": event_time,
                            "event_type": random.choice(event_types),
                            "page_url": random.choice(["/home", "/products", "/categories", "/deals"]),
                            "duration_seconds": random.randint(10, 300),
                        }
                    )

        self.events_df = pd.DataFrame(events)
        return self.events_df

    def generate_transactions(self) -> pd.DataFrame:
        logger.info("Generando transacciones")
        if self.customers_df is None:
            raise ValueError("Primero genera los perfiles de clientes")

        transactions = []

        buying_customers = random.sample(list(self.customers_df["customer_id"]), int(self.n_customers * 0.6))

        for customer_id in buying_customers:
            n_purchases = random.randint(1, 8)

            for purchase in range(n_purchases):
                purchase_date = datetime.now() - timedelta(days=random.randint(1, 180))

                transactions.append(
                    {
                        "transaction_id": f"TRX_{customer_id}_{purchase}",
                        "customer_id": customer_id,
                        "timestamp": purchase_date,
                        "amount": round(random.uniform(15000, 250000), -3),
                        "items_count": random.randint(1, 5),
                        "payment_method": random.choice(["credit_card", "debit_card", "pse", "cash"]),
                    }
                )

        self.transactions_df = pd.DataFrame(transactions)
        return self.transactions_df

    def generate_churn_labels(self, churn_threshold_days: int = 60) -> pd.DataFrame:
        logger.info("Generando etiquetas de churn")
        if self.events_df is None:
            raise ValueError("Primero genera los eventos")

        last_activity = self.events_df.groupby("customer_id")["timestamp"].max()
        analysis_date = self.events_df["timestamp"].max()

        churn_labels = []
        for customer_id, last_active in last_activity.items():
            days_inactive = (analysis_date - last_active).days
            is_churn = 1 if days_inactive > churn_threshold_days else 0

            churn_labels.append(
                {
                    "customer_id": customer_id,
                    "churn": is_churn,
                    "days_since_last_activity": days_inactive,
                }
            )

        return pd.DataFrame(churn_labels)

    def generate_all_data(self) -> GeneratedData:
        logger.info("Iniciando generacion de datos completos")
        customers = self.generate_customer_profiles()
        events = self.generate_behavioral_data()
        transactions = self.generate_transactions()
        churn_labels = self.generate_churn_labels()

        customers.to_csv(paths.data_raw / "customers.csv", index=False)
        events.to_csv(paths.data_raw / "events.csv", index=False)
        transactions.to_csv(paths.data_raw / "transactions.csv", index=False)
        churn_labels.to_csv(paths.data_raw / "churn_labels.csv", index=False)

        logger.info(
            "Datos generados: %d clientes, %d eventos, %d transacciones", len(customers), len(events), len(transactions)
        )
        logger.info("Tasa de churn: %.1f%%", churn_labels["churn"].mean() * 100)

        return GeneratedData(customers, events, transactions, churn_labels)


if __name__ == "__main__":
    from churn_microstores.logging_utils import configure_logging

    configure_logging()
    generator = DataGenerator(n_customers=generation.n_customers)
    generated = generator.generate_all_data()
    logger.info("Muestra de datos:\n%s", generated.customers.head())
