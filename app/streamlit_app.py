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
    get_alert_customers,
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
    st.metric("Tasa de churn", f"{churn_rate:.1%}", help="Proporcion de clientes etiquetados como churn")
    st.metric("Gasto promedio", f"${avg_spent:,.0f}", help="Monto promedio gastado por cliente")


def scenario_explorer(predictor: ChurnPredictor, features_df: pd.DataFrame) -> None:
    st.subheader("Explorador de escenarios")
    st.markdown("Ajusta las variables clave para estimar la probabilidad de churn de un cliente hipotetico.")
    col1, col2, col3 = st.columns(3)

    total_spent = col1.slider("Total gastado", 0, int(features_df["total_spent"].max()), 50000, step=5000)
    days_inactive = col2.slider(
        "Dias desde ultima actividad", 0, int(features_df["days_since_last_activity"].max()), 15, step=5
    )
    sessions = col3.slider("Total de sesiones", 0, int(features_df["total_sessions"].max()), 5, step=1)

    synthetic_row = features_df.median(numeric_only=True)
    synthetic_row["total_spent"] = total_spent
    synthetic_row["days_since_last_activity"] = days_inactive
    synthetic_row["total_sessions"] = sessions

    features_for_model = synthetic_row.drop(labels=["customer_id"], errors="ignore")
    _, probability = predictor.predict_churn(features_for_model)

    st.write(f"Probabilidad estimada de churn: {probability:.1%}")

    st.write("Recomendaciones sugeridas:")
    if probability >= 0.7:
        st.markdown("- Riesgo alto: contactar de forma proactiva y ofrecer un incentivo fuerte de retencion.")
        st.markdown("- Revisar si hubo cambios recientes en precios, tiempos de entrega o experiencia.")
    elif probability >= 0.4:
        st.markdown("- Riesgo moderado: reforzar comunicacion y valor percibido.")
        st.markdown("- Enviar contenido relevante, recomendaciones y recordatorios.")
    else:
        st.markdown("- Riesgo bajo: mantener buen servicio y programas de fidelizacion.")
        st.markdown("- Es un buen candidato para propuestas de upsell controlado.")


st.title("Sistema de Prevencion de Churn para Microstores")
st.markdown(
    """
    Esta aplicacion muestra un pipeline completo de generacion de datos sinteticos, entrenamiento de modelos y analisis de riesgo
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
    st.header("Resumen rapido")
    render_summary_cards(features_df, churn_labels)

# Gestion de datos: elegir fuente y permitir agregar clientes
st.header("1. Gestion de datos de clientes")

if "working_features_df" not in st.session_state:
    st.session_state["working_features_df"] = features_df.copy()
if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "Datos sinteticos generados"

data_mode = st.radio(
    "¿Que datos quieres usar para el analisis?",
    options=["Datos sinteticos generados", "Subir mi propia base de clientes"],
    index=0 if st.session_state["data_mode"] == "Datos sinteticos generados" else 1,
)
st.session_state["data_mode"] = data_mode

uploaded_df = None
if data_mode == "Subir mi propia base de clientes":
    st.markdown(
        "Sube un archivo CSV con al menos estas columnas: "
        "`customer_id`, `total_spent`, `days_since_last_activity`, `total_sessions`."
    )
    file = st.file_uploader("Sube tu base de clientes", type=["csv"])
    if file is not None:
        uploaded_df = pd.read_csv(file)
        st.write("Vista previa de los datos subidos:")
        st.dataframe(uploaded_df.head())

        required_cols = ["customer_id", "total_spent", "days_since_last_activity", "total_sessions"]
        missing = [c for c in required_cols if c not in uploaded_df.columns]
        if missing:
            st.error(f"Faltan columnas requeridas: {missing}")
            st.stop()

        st.session_state["working_features_df"] = uploaded_df.copy()
else:
    st.session_state["working_features_df"] = features_df.copy()

st.subheader("Agregar un cliente manualmente")

with st.form("nuevo_cliente_form"):
    new_id = st.text_input("ID de cliente")
    new_total_spent = st.number_input("Total gastado", min_value=0.0, step=1000.0)
    new_days_inactive = st.number_input("Dias desde ultima actividad", min_value=0, step=1)
    new_sessions = st.number_input("Total de sesiones", min_value=0, step=1)

    submitted = st.form_submit_button("Agregar cliente")

if submitted:
    if not new_id:
        st.error("Debes proporcionar un ID de cliente.")
    else:
        base = st.session_state["working_features_df"].median(numeric_only=True).to_dict()
        new_row = {
            "customer_id": new_id,
            "total_spent": new_total_spent,
            "days_since_last_activity": new_days_inactive,
            "total_sessions": new_sessions,
        }
        base.update(new_row)
        new_row_df = pd.DataFrame([base])
        st.session_state["working_features_df"] = pd.concat(
            [st.session_state["working_features_df"], new_row_df],
            ignore_index=True,
        )
        st.success(f"Cliente {new_id} agregado a la base de trabajo en esta sesion.")

current_features = st.session_state["working_features_df"]

# Alertas de negocio
st.header("2. Alertas y salud del negocio")

if churn_labels is not None and "customer_id" in churn_labels.columns:
    analysis_df = current_features.merge(churn_labels, on="customer_id", how="left")
else:
    analysis_df = current_features.copy()

alerts = generate_alerts(analysis_df)
alert_customers = get_alert_customers(analysis_df)

if alerts is not None and not alerts.empty:
    for _, row in alerts.iterrows():
        segment = row.get("segment", "segmento_desconocido")
        num_customers = row.get("num_customers", 0)
        description = row.get("description", "")
        recommended_action = row.get("recommended_action", "")

        with st.expander(f"{segment} ({num_customers} clientes)"):
            if description:
                st.write(description)
            if recommended_action:
                st.markdown(f"**Accion recomendada:** {recommended_action}")

            seg_df = alert_customers.get(segment)
            if seg_df is not None and not seg_df.empty:
                st.markdown("Clientes en este segmento:")
                cols_priority = ["customer_id", "total_spent", "days_since_last_activity", "purchase_frequency"]
                churn_cols = [c for c in seg_df.columns if c.startswith("churn")]
                selected_cols = [c for c in cols_priority if c in seg_df.columns] + churn_cols
                if not selected_cols:
                    selected_cols = seg_df.columns.tolist()
                st.dataframe(seg_df[selected_cols])
else:
    st.success("Sin alertas criticas con los datos actuales.")

# Segmentos clave
st.header("3. Segmentos clave")

if churn_labels is not None and "customer_id" in churn_labels.columns:
    segments_df = analyze_customer_segments(current_features, churn_labels)
else:
    segments_df = analyze_customer_segments(current_features, None)

st.dataframe(segments_df)

# Recomendaciones por cliente (reglas de comportamiento sobre la base actual)
if churn_labels is not None and "customer_id" in churn_labels.columns:
    recs_df = generate_recommendations(current_features, churn_labels)
else:
    recs_df = generate_recommendations(current_features, None)

# Explorador de escenarios
st.header("4. Explorador de escenarios")

if st.session_state["data_mode"] == "Datos sinteticos generados":
    scenario_explorer(predictor, current_features)
else:
    st.info(
        "El explorador de escenarios usa el modelo entrenado con datos sinteticos. "
        "Selecciona 'Datos sinteticos generados' para activarlo."
    )

# Distribuciones principales
st.header("5. Distribuciones principales")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Frecuencia de sesiones por cliente ayuda a identificar engagement general.")
    fig, ax = plt.subplots()
    sns.histplot(current_features["total_sessions"], ax=ax)
    st.pyplot(fig)
with col2:
    st.markdown("Gasto total por cliente muestra el valor economico de la base.")
    fig, ax = plt.subplots()
    sns.histplot(current_features["total_spent"], ax=ax)
    st.pyplot(fig)

# Recomendaciones personalizadas
st.header("6. Recomendaciones personalizadas")

customer_id = st.selectbox("Selecciona un cliente", options=current_features["customer_id"])
selected_row = current_features[current_features["customer_id"] == customer_id].iloc[0]

if st.session_state["data_mode"] == "Datos sinteticos generados":
    features_for_model = selected_row.drop(labels=["customer_id"], errors="ignore")
    _, probability = predictor.predict_churn(features_for_model)
    st.write(f"Probabilidad de churn para {customer_id}: {probability:.1%}")
else:
    probability = None
    st.info("Probabilidad de churn disponible solo con datos sinteticos generados.")

customer_recs = recs_df[recs_df["customer_id"] == customer_id] if not recs_df.empty else pd.DataFrame()

if not customer_recs.empty:
    st.write("Recomendaciones sugeridas:")
    for _, row in customer_recs.iterrows():
        action = row.get("recommended_action", "")
        segment = row.get("segment", "")
        reason = row.get("reason", "")
        if action:
            st.markdown(f"- {action} ({segment})")
        if reason:
            st.caption(reason)
else:
    st.info("No se encontraron recomendaciones especificas para este cliente.")
