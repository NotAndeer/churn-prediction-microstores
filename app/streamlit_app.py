from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from churn_microstores.config import paths
from churn_microstores.io_utils import load_features, load_labels
from churn_microstores.logging_utils import configure_logging
from churn_microstores.models import ChurnPredictor
from churn_microstores.recommendations import (
    analyze_customer_segments,
    generate_alerts,
    generate_recommendations,
)

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sistema de Prevencion de Churn", layout="wide")


@st.cache_data
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers_df = pd.read_csv(paths.data_raw / "customers.csv", parse_dates=["signup_date"])
    events_df = pd.read_csv(paths.data_raw / "events.csv", parse_dates=["timestamp"])
    transactions_df = pd.read_csv(paths.data_raw / "transactions.csv", parse_dates=["timestamp"])
    features_df = load_features(paths.data_processed / "features.csv")
    churn_labels = load_labels(paths.data_raw / "churn_labels.csv")
    return customers_df, events_df, transactions_df, features_df, churn_labels


@st.cache_resource
def load_predictor(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> ChurnPredictor:
    predictor = ChurnPredictor()
    X_train, X_test, y_train, y_test, _ = predictor.prepare_data(
        paths.data_processed / "features.csv", paths.data_raw / "churn_labels.csv"
    )
    predictor.train_models(X_train, X_test, y_train, y_test)
    return predictor


def render_summary_cards(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    churn_rate = labels_df["churn"].mean()
    total_customers = len(features_df)
    avg_spent = features_df["total_spent"].mean()

    st.metric("Clientes", total_customers, help="Total de clientes analizados")
    st.metric("Tasa de churn", f"{churn_rate:.1%}", help="Proporción de clientes etiquetados como churn")
    st.metric("Gasto promedio", f"${avg_spent:,.0f}", help="Monto promedio gastado por cliente")


def scenario_explorer(predictor: ChurnPredictor, features_df: pd.DataFrame) -> None:
    st.subheader("Scenario explorer")
    st.markdown("Ajusta las variables clave para estimar la probabilidad de churn de un cliente hipotético.")
    col1, col2, col3 = st.columns(3)

    total_spent = col1.slider("Total gastado", 0, int(features_df["total_spent"].max()), 50000, step=5000)
    days_inactive = col2.slider(
        "Días desde última actividad", 0, int(features_df["days_since_last_activity"].max()), 15, step=5
    )
    sessions = col3.slider("Total de sesiones", 0, int(features_df["total_sessions"].max()), 5, step=1)

    synthetic_row = features_df.median(numeric_only=True)
    synthetic_row["total_spent"] = total_spent
    synthetic_row["days_since_last_activity"] = days_inactive
    synthetic_row["total_sessions"] = sessions

    prediction, probability = predictor.predict_churn(synthetic_row)
    st.write(f"Probabilidad estimada de churn: **{probability:.1%}**")
    recs, _ = generate_recommendations(synthetic_row, predictor.models[predictor.best_model_name or "xgboost"])
    st.write("Recomendaciones sugeridas:")
    for rec in recs:
        st.markdown(f"- {rec}")


st.title("Sistema de Prevención de Churn para Microstores")
st.markdown(
    """
    Esta aplicación muestra un pipeline completo de generación de datos sintéticos, entrenamiento de modelos y análisis de riesgo
    de churn para microtiendas. Usa los controles para explorar el riesgo y las recomendaciones.
    """
)

try:
    customers_df, events_df, transactions_df, features_df, churn_labels = load_all_data()
    predictor = load_predictor(features_df, churn_labels)
except FileNotFoundError:
    st.error("No se encontraron datos procesados. Ejecuta `python reproduce.py` antes de abrir la app.")
    st.stop()

with st.sidebar:
    st.header("Resumen rápido")
    render_summary_cards(features_df, churn_labels)

st.header("Alertas y salud del negocio")
alerts = generate_alerts(features_df.merge(churn_labels, on="customer_id"))
if alerts:
    for message, level, _ in alerts:
        st.info(message) if level == "info" else st.warning(message) if level == "warning" else st.error(message)
else:
    st.success("Sin alertas críticas")

st.header("Segmentos clave")
feature_matrix = features_df.drop(columns=["customer_id"])
predictions = predictor.models[predictor.best_model_name or "xgboost"].predict_proba(feature_matrix)[:, 1]
segments_df = analyze_customer_segments(features_df, predictions)
st.dataframe(segments_df)

scenario_explorer(predictor, features_df)

st.header("Distribuciones principales")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Frecuencia de sesiones por cliente ayuda a identificar engagement general.")
    fig, ax = plt.subplots()
    sns.histplot(features_df["total_sessions"], ax=ax)
    st.pyplot(fig)
with col2:
    st.markdown("Gasto total por cliente muestra el valor económico de la base.")
    fig, ax = plt.subplots()
    sns.histplot(features_df["total_spent"], ax=ax)
    st.pyplot(fig)

st.header("Recomendaciones personalizadas")
customer_id = st.selectbox("Selecciona un cliente", options=features_df["customer_id"])
selected_row = features_df[features_df["customer_id"] == customer_id].iloc[0]
prediction, probability = predictor.predict_churn(selected_row.drop("customer_id"))
recs, probability = generate_recommendations(
    selected_row.drop("customer_id"), predictor.models[predictor.best_model_name or "xgboost"]
)

st.write(f"Probabilidad de churn para {customer_id}: {probability:.1%}")
for rec in recs:
    st.markdown(f"- {rec}")
