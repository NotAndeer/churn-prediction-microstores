"""Pipeline runner for churn microstores."""
from __future__ import annotations

import logging

from churn_microstores.config import paths
from churn_microstores.data_generation import DataGenerator
from churn_microstores.features import create_features
from churn_microstores.logging_utils import configure_logging
from churn_microstores.models import run_model_pipeline


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando pipeline completo del sistema predictivo")

    generator = DataGenerator()
    generator.generate_all_data()

    create_features(
        paths.data_raw / "customers.csv",
        paths.data_raw / "events.csv",
        paths.data_raw / "transactions.csv",
        paths.data_processed / "features.csv",
    )

    run_model_pipeline()
    logger.info("Pipeline completado exitosamente")
    logger.info("Para iniciar el dashboard ejecuta: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
