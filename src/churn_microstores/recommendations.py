from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_recommendations(customer_data: pd.Series, model, risk_threshold: float = 0.7) -> Tuple[list[str], float]:
    """Genera recomendaciones basadas en los datos del cliente y predicción del modelo."""

    probability = float(model.predict_proba(customer_data.values.reshape(1, -1))[0, 1])

    if probability < risk_threshold:
        return ["Cliente de bajo riesgo - Mantener engagement natural"], probability

    recommendations: list[str] = []

    if "days_since_last_activity" in customer_data.index:
        days_inactive = customer_data["days_since_last_activity"]
        if days_inactive > 30:
            recommendations.append(
                f"Cliente inactivo por {int(days_inactive)} dias. Enviar email de reactivacion con cupon del 15%"
            )

    if "cart_additions" in customer_data.index:
        carts = customer_data["cart_additions"]
        if carts > 0:
            recommendations.append(f"Tiene {int(carts)} productos en carrito. Ofrecer envio gratis en esos productos")

    if "total_purchases" in customer_data.index:
        purchases = customer_data["total_purchases"]
        if purchases == 0:
            recommendations.append("Cliente nunca ha comprado. Ofrecer descuento de primer compra")
        elif purchases > 3:
            recommendations.append(f"Cliente frecuente ({int(purchases)} compras). Ofrecer programa de fidelizacion")

    if "avg_session_duration" in customer_data.index:
        session_duration = customer_data["avg_session_duration"]
        if session_duration < 60:
            recommendations.append(f"Sesiones muy cortas ({session_duration:.1f} segundos). Mejorar experiencia de usuario")

    if not recommendations:
        recommendations.append("Revisar manualmente - Caso complejo")

    logger.debug("Recomendaciones generadas: %s", recommendations)
    return recommendations, probability


def analyze_customer_segments(features_df: pd.DataFrame, churn_predictions: np.ndarray) -> pd.DataFrame:
    """Analiza segmentos de clientes para estrategias de retencion."""

    data = features_df.copy()
    data["churn_probability"] = churn_predictions

    segments = []

    inactive_customers = data[data["days_since_last_activity"] > 45]
    if len(inactive_customers) > 0:
        segments.append(
            {
                "segmento": "Clientes Inactivos",
                "tamano": len(inactive_customers),
                "riesgo_promedio": inactive_customers["churn_probability"].mean(),
                "estrategia": "Campaña de reactivacion con incentivos",
            }
        )

    cart_abandoners = data[data["cart_additions"] > 0]
    if len(cart_abandoners) > 0:
        segments.append(
            {
                "segmento": "Abandonan Carritos",
                "tamano": len(cart_abandoners),
                "riesgo_promedio": cart_abandoners["churn_probability"].mean(),
                "estrategia": "Recordatorios y ofertas en productos del carrito",
            }
        )

    frequent_buyers = data[data["total_purchases"] > 2]
    high_risk_frequent = frequent_buyers[frequent_buyers["churn_probability"] > 0.6]
    if len(high_risk_frequent) > 0:
        segments.append(
            {
                "segmento": "Compradores Frecuentes en Riesgo",
                "tamano": len(high_risk_frequent),
                "riesgo_promedio": high_risk_frequent["churn_probability"].mean(),
                "estrategia": "Programa de fidelizacion premium",
            }
        )

    return pd.DataFrame(segments)


def generate_alerts(analysis_df: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Genera alertas de negocio basadas en métricas clave."""

    alertas: list[tuple[str, str, float]] = []
    churn_rate = analysis_df["churn"].mean()
    if churn_rate > 0.15:
        alertas.append((f"Tasa de churn CRITICA: {churn_rate:.1%} (>15%)", "error", float(churn_rate)))
    elif churn_rate > 0.10:
        alertas.append((f"Tasa de churn ALTA: {churn_rate:.1%} (>10%)", "warning", float(churn_rate)))

    clientes_valiosos_inactivos = analysis_df[
        (analysis_df["days_since_last_activity"] > 30)
        & (analysis_df["total_spent"] > analysis_df["total_spent"].quantile(0.7))
    ]
    if len(clientes_valiosos_inactivos) > 10:
        alertas.append((f"{len(clientes_valiosos_inactivos)} clientes valiosos inactivos >30 días", "error", float(len(clientes_valiosos_inactivos))))
    elif len(clientes_valiosos_inactivos) > 5:
        alertas.append((f"{len(clientes_valiosos_inactivos)} clientes valiosos inactivos >30 días", "warning", float(len(clientes_valiosos_inactivos))))

    avg_conversion = analysis_df["purchase_conversion_rate"].mean()
    if avg_conversion < 0.1:
        alertas.append((f"Conversion BAJA: {avg_conversion:.1%} (<10%)", "warning", float(avg_conversion)))

    top_10_clientes = analysis_df.nlargest(10, "total_spent")
    concentracion_riesgo = top_10_clientes["churn"].sum()
    if concentracion_riesgo >= 3:
        alertas.append((
            f"ALTO RIESGO: {concentracion_riesgo} de top 10 clientes podrían abandonar",
            "error",
            float(concentracion_riesgo),
        ))

    avg_sessions = analysis_df["total_sessions"].mean()
    if avg_sessions < 5:
        alertas.append((f"Engagement BAJO: {avg_sessions:.1f} sesiones promedio por cliente", "warning", float(avg_sessions)))

    avg_cart_abandonment = analysis_df["cart_addition_rate"].mean()
    if avg_cart_abandonment > 0.8:
        alertas.append((f"Alta tasa carritos abandonados: {avg_cart_abandonment:.1%}", "warning", float(avg_cart_abandonment)))

    if churn_rate < 0.05:
        alertas.append((f"EXCELENTE: Tasa de churn baja {churn_rate:.1%}", "info", float(churn_rate)))

    return alertas


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Modulo de recomendaciones cargado correctamente")
