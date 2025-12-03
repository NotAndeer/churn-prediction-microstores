import logging
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas mínimas que esperamos para poder generar análisis y alertas
REQUIRED_BASE_COLUMNS: List[str] = [
    "customer_id",
    "days_since_last_activity",
]


def _normalize_days_since_last_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame tenga una columna llamada 'days_since_last_activity'.

    Cuando se hace un merge, es común que pandas genere columnas con sufijos
    como '_x' y '_y'. Esta función busca variantes conocidas y las normaliza
    copiándolas a 'days_since_last_activity'.

    Si no se encuentra ninguna variante, lanza un ValueError con información clara.
    """
    candidates = [
        "days_since_last_activity",
        "days_since_last_activity_x",
        "days_since_last_activity_y",
    ]

    for col in candidates:
        if col in df.columns:
            if col != "days_since_last_activity":
                logger.warning(
                    "Normalizando columna '%s' -> 'days_since_last_activity' para analisis",
                    col,
                )
                df["days_since_last_activity"] = df[col]
            return df

    raise ValueError(
        "No se encontró la columna 'days_since_last_activity' ni variantes "
        f"en el dataframe. Columnas disponibles: {list(df.columns)}"
    )


def _get_churn_column(df: pd.DataFrame) -> Optional[str]:
    """
    Devuelve el nombre de la columna que representa churn o probabilidad de churn,
    si existe alguna. Esto hace que el módulo sea más tolerante a cambios de nombre.
    """
    candidates = [
        "churn_probability",
        "churn_prob",
        "churn_score",
        "churn",
        "churn_label",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Valida que el dataframe tenga al menos las columnas base necesarias.

    Si faltan, lanza un ValueError con el detalle de qué falta y qué columnas hay.
    """
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas para analisis/alertas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def _prepare_alert_base_df(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str], dict]:
    """
    Normaliza y prepara el dataframe base para generar alertas y segmentaciones,
    devolviendo:
      - df limpio
      - nombre de la columna de churn (si existe)
      - diccionario con umbrales usados
    """
    if analysis_df.empty:
        return analysis_df.copy(), None, {}

    df = analysis_df.copy()
    df = _normalize_days_since_last_activity(df)
    _validate_required_columns(df)

    if "total_spent" not in df.columns:
        logger.warning("Columna 'total_spent' no encontrada; se rellena con 0 para analisis/alertas basicos")
        df["total_spent"] = 0.0

    if "purchase_frequency" not in df.columns:
        logger.warning(
            "Columna 'purchase_frequency' no encontrada; se rellena con 0 para analisis/alertas basicos"
        )
        df["purchase_frequency"] = 0.0

    churn_col = _get_churn_column(df)

    thresholds = {}
    thresholds["inactivity_threshold"] = 30
    thresholds["high_value_threshold"] = df["total_spent"].quantile(0.75)
    thresholds["low_value_threshold"] = df["total_spent"].quantile(0.25)
    thresholds["high_frequency_threshold"] = df["purchase_frequency"].quantile(0.75)

    return df, churn_col, thresholds


def _compute_alert_segments(df: pd.DataFrame, churn_col: Optional[str], thresholds: dict) -> tuple[list, Dict[str, pd.Series]]:
    """
    A partir de un df ya normalizado y umbrales calculados, construye:
      - lista de dicts de alertas agregadas
      - diccionario segment_name -> mascara booleana de clientes que pertenecen a ese segmento

    Esto se reutiliza tanto para generate_alerts (vista ejecutiva) como para
    get_alert_customers (detalle por cliente).
    """
    if df.empty:
        return [], {}

    inactivity_threshold = thresholds["inactivity_threshold"]
    high_value_threshold = thresholds["high_value_threshold"]
    low_value_threshold = thresholds["low_value_threshold"]
    high_frequency_threshold = thresholds["high_frequency_threshold"]

    inactive_mask = df["days_since_last_activity"] > inactivity_threshold
    active_mask = ~inactive_mask

    high_value_mask = df["total_spent"] >= high_value_threshold
    low_value_mask = df["total_spent"] <= low_value_threshold
    high_freq_mask = df["purchase_frequency"] >= high_frequency_threshold

    alerts = []
    segment_masks: Dict[str, pd.Series] = {}

    # Segmento: clientes inactivos de alto valor
    segment_name = "inactive_high_value"
    segment_mask = inactive_mask & high_value_mask
    segment_masks[segment_name] = segment_mask
    num_customers = int(segment_mask.sum())
    if num_customers > 0:
        alerts.append(
            {
                "segment": segment_name,
                "num_customers": num_customers,
                "description": (
                    "Clientes con alto valor total pero inactivos durante mas de "
                    f"{inactivity_threshold} dias."
                ),
                "recommended_action": (
                    "Lanzar campaña personalizada de recuperacion (descuentos exclusivos, "
                    "contacto directo por canales directos)."
                ),
            }
        )

    # Segmento: clientes inactivos de bajo valor
    segment_name = "inactive_low_value"
    segment_mask = inactive_mask & low_value_mask
    segment_masks[segment_name] = segment_mask
    num_customers = int(segment_mask.sum())
    if num_customers > 0:
        alerts.append(
            {
                "segment": segment_name,
                "num_customers": num_customers,
                "description": (
                    "Clientes de bajo valor que llevan mas de "
                    f"{inactivity_threshold} dias sin actividad."
                ),
                "recommended_action": (
                    "Automatizar recordatorios ligeros (newsletter, notificaciones) "
                    "y evaluar si vale la pena invertir mas en este segmento."
                ),
            }
        )

    # Segmento: clientes activos con alta frecuencia
    segment_name = "active_high_frequency"
    segment_mask = active_mask & high_freq_mask
    segment_masks[segment_name] = segment_mask
    num_customers = int(segment_mask.sum())
    if num_customers > 0:
        alerts.append(
            {
                "segment": segment_name,
                "num_customers": num_customers,
                "description": (
                    "Clientes activos que compran con alta frecuencia. Segmento ideal "
                    "para programas de fidelizacion y upsell."
                ),
                "recommended_action": (
                    "Ofrecer programas de lealtad, paquetes premium o productos "
                    "complementarios basados en su historial."
                ),
            }
        )

    # Segmento: clientes nuevos con riesgo de enfriarse
    if "days_since_signup" in df.columns:
        segment_name = "new_customers_at_risk"
        new_customer_threshold = df["days_since_signup"].quantile(0.25)
        new_customers_mask = df["days_since_signup"] <= new_customer_threshold
        segment_mask = new_customers_mask & inactive_mask
        segment_masks[segment_name] = segment_mask
        num_customers = int(segment_mask.sum())
        if num_customers > 0:
            alerts.append(
                {
                    "segment": segment_name,
                    "num_customers": num_customers,
                    "description": (
                        "Clientes nuevos que muestran poca actividad y podrian abandonar "
                        "antes de consolidarse."
                    ),
                    "recommended_action": (
                        "Refinar el onboarding: mensajes guiados, recomendaciones "
                        "personalizadas y ofertas para primeras compras."
                    ),
                }
            )

    # Segmento: alto riesgo de churn segun modelo
    if churn_col is not None:
        segment_name = "high_churn_risk"
        churn_values = df[churn_col].astype(float)
        unique_values = sorted(churn_values.dropna().unique())
        if unique_values and set(unique_values).issubset({0.0, 1.0}):
            high_risk_mask = churn_values == 1.0
        else:
            high_risk_threshold = 0.7
            high_risk_mask = churn_values >= high_risk_threshold

        segment_masks[segment_name] = high_risk_mask
        num_customers = int(high_risk_mask.sum())
        if num_customers > 0:
            alerts.append(
                {
                    "segment": segment_name,
                    "num_customers": num_customers,
                    "description": (
                        "Clientes identificados con alto riesgo de abandono "
                        f"segun la columna '{churn_col}'."
                    ),
                    "recommended_action": (
                        "Priorizar contacto proactivo, encuestas de satisfaccion y "
                        "ofertas especificas para retencion."
                    ),
                }
            )

    return alerts, segment_masks


def analyze_customer_segments(
    features_df: pd.DataFrame,
    churn_labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Analiza segmentos de clientes a partir de las features y, opcionalmente,
    etiquetas o probabilidades de churn.

    Devuelve un DataFrame con métricas agregadas por segmento.
    """
    if features_df.empty:
        logger.warning("analyze_customer_segments recibio un dataframe vacio; no se generara analisis")
        return pd.DataFrame(
            columns=[
                "segment",
                "num_customers",
                "avg_total_spent",
                "avg_days_since_last_activity",
                "avg_purchase_frequency",
                "churn_rate",
                "description",
            ]
        )

    if churn_labels is not None and "customer_id" in churn_labels.columns:
        df = features_df.merge(churn_labels, on="customer_id", how="left")
    else:
        df = features_df.copy()

    df = _normalize_days_since_last_activity(df)
    _validate_required_columns(df)

    if "total_spent" not in df.columns:
        df["total_spent"] = 0.0

    if "purchase_frequency" not in df.columns:
        df["purchase_frequency"] = 0.0

    churn_col = _get_churn_column(df)

    inactivity_threshold = 30
    high_value_threshold = df["total_spent"].quantile(0.75)
    low_value_threshold = df["total_spent"].quantile(0.25)
    high_frequency_threshold = df["purchase_frequency"].quantile(0.75)

    inactive_mask = df["days_since_last_activity"] > inactivity_threshold
    active_mask = ~inactive_mask

    high_value_mask = df["total_spent"] >= high_value_threshold
    low_value_mask = df["total_spent"] <= low_value_threshold
    high_freq_mask = df["purchase_frequency"] >= high_frequency_threshold

    segments = []

    def add_segment(
        name: str,
        mask: pd.Series,
        description: str,
    ) -> None:
        segment_df = df[mask]
        num_customers = int(mask.sum())
        if num_customers == 0:
            return

        churn_rate = np.nan
        if churn_col is not None:
            churn_values = segment_df[churn_col].astype(float)
            unique_values = sorted(churn_values.dropna().unique())
            if unique_values and set(unique_values).issubset({0.0, 1.0}):
                churn_rate = float(churn_values.mean())
            elif not churn_values.empty:
                churn_rate = float((churn_values >= 0.7).mean())

        segments.append(
            {
                "segment": name,
                "num_customers": num_customers,
                "avg_total_spent": float(segment_df["total_spent"].mean()),
                "avg_days_since_last_activity": float(
                    segment_df["days_since_last_activity"].mean()
                ),
                "avg_purchase_frequency": float(segment_df["purchase_frequency"].mean()),
                "churn_rate": churn_rate,
                "description": description,
            }
        )

    add_segment(
        name="inactive_high_value",
        mask=inactive_mask & high_value_mask,
        description=(
            "Clientes inactivos (mas de 30 dias) con alto valor de compra acumulado."
        ),
    )

    add_segment(
        name="inactive_low_value",
        mask=inactive_mask & low_value_mask,
        description="Clientes inactivos (mas de 30 dias) con bajo valor acumulado.",
    )

    add_segment(
        name="active_high_frequency",
        mask=active_mask & high_freq_mask,
        description="Clientes activos con alta frecuencia de compra.",
    )

    add_segment(
        name="active_low_value",
        mask=active_mask & low_value_mask,
        description="Clientes activos pero de bajo valor acumulado.",
    )

    if "days_since_signup" in df.columns:
        new_customer_threshold = df["days_since_signup"].quantile(0.25)
        new_customers_mask = df["days_since_signup"] <= new_customer_threshold
        add_segment(
            name="new_customers",
            mask=new_customers_mask,
            description="Clientes relativamente nuevos desde su fecha de alta.",
        )

    if not segments:
        logger.info("No se encontraron segmentos con condiciones suficientes para analisis.")
        return pd.DataFrame(
            [
                {
                    "segment": "no_segments",
                    "num_customers": 0,
                    "avg_total_spent": 0.0,
                    "avg_days_since_last_activity": 0.0,
                    "avg_purchase_frequency": 0.0,
                    "churn_rate": np.nan,
                    "description": "No se detectaron segmentos diferenciados con los datos actuales.",
                }
            ]
        )

    segments_df = pd.DataFrame(segments)
    segments_df = segments_df.sort_values(by="num_customers", ascending=False).reset_index(drop=True)

    logger.info("Analisis de segmentos generado: %d segmentos", len(segments_df))
    return segments_df


def generate_recommendations(
    features_df: pd.DataFrame,
    churn_labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Genera recomendaciones a nivel cliente, basadas en su actividad, valor y (si existe)
    informacion de churn.

    Devuelve un DataFrame con columnas:
        - customer_id
        - segment
        - recommended_action
        - reason
    """
    if features_df.empty:
        logger.warning("generate_recommendations recibio un dataframe vacio; no se generaran recomendaciones")
        return pd.DataFrame(
            columns=[
                "customer_id",
                "segment",
                "recommended_action",
                "reason",
            ]
        )

    if churn_labels is not None and "customer_id" in churn_labels.columns:
        df = features_df.merge(churn_labels, on="customer_id", how="left")
    else:
        df = features_df.copy()

    df, churn_col, thresholds = _prepare_alert_base_df(df)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "segment",
                "recommended_action",
                "reason",
            ]
        )

    inactivity_threshold = thresholds["inactivity_threshold"]
    high_value_threshold = thresholds["high_value_threshold"]
    low_value_threshold = thresholds["low_value_threshold"]
    high_frequency_threshold = thresholds["high_frequency_threshold"]

    inactive_mask = df["days_since_last_activity"] > inactivity_threshold
    active_mask = ~inactive_mask

    high_value_mask = df["total_spent"] >= high_value_threshold
    low_value_mask = df["total_spent"] <= low_value_threshold
    high_freq_mask = df["purchase_frequency"] >= high_frequency_threshold

    segments = []

    def add_recommendation(mask: pd.Series, segment: str, action: str, reason_builder) -> None:
        segment_df = df[mask].copy()
        if segment_df.empty:
            return
        segment_df["segment"] = segment
        segment_df["recommended_action"] = action
        segment_df["reason"] = segment_df.apply(reason_builder, axis=1)
        segments.append(
            segment_df[["customer_id", "segment", "recommended_action", "reason"]]
        )

    # Clientes valiosos e inactivos
    add_recommendation(
        mask=inactive_mask & high_value_mask,
        segment="inactive_high_value",
        action="Lanzar campaña personalizada de reactivacion con oferta fuerte y contacto directo.",
        reason_builder=lambda row: (
            f"Este cliente ha gastado aproximadamente {row['total_spent']:.0f} unidades en total, "
            f"pero lleva {int(row['days_since_last_activity'])} dias sin actividad. "
            "Historicamente ha demostrado alto valor para el negocio, por lo que perderlo "
            "implicaria un impacto importante en ingresos."
        ),
    )

    # Clientes de bajo valor e inactivos
    add_recommendation(
        mask=inactive_mask & low_value_mask,
        segment="inactive_low_value",
        action="Enviar recordatorios automatizados de bajo costo y evaluar si vale la pena invertir mas.",
        reason_builder=lambda row: (
            f"Este cliente tiene un gasto acumulado relativamente bajo (alrededor de {row['total_spent']:.0f} unidades) "
            f"y lleva {int(row['days_since_last_activity'])} dias sin actividad. "
            "Es poco probable que una inversion fuerte en retencion tenga buen retorno."
        ),
    )

    # Clientes activos y de alta frecuencia
    add_recommendation(
        mask=active_mask & high_freq_mask,
        segment="active_high_frequency",
        action="Ofrecer programa de fidelizacion y productos complementarios o premium.",
        reason_builder=lambda row: (
            f"Este cliente interactua con frecuencia (frecuencia de compra aproximada {row['purchase_frequency']:.2f}) "
            "y se mantiene activo recientemente. Es un perfil ideal para reforzar lealtad y explorar ventas adicionales."
        ),
    )

    # Clientes nuevos con riesgo de enfriarse
    if "days_since_signup" in df.columns:
        new_customer_threshold = df["days_since_signup"].quantile(0.25)
        new_customers_mask = df["days_since_signup"] <= new_customer_threshold

        add_recommendation(
            mask=new_customers_mask & inactive_mask,
            segment="new_customers_at_risk",
            action="Refinar onboarding y ofrecer incentivos claros para primeras compras.",
            reason_builder=lambda row: (
                f"Este cliente lleva poco tiempo desde su registro (alrededor de {int(row['days_since_signup'])} dias), "
                f"pero ya tiene {int(row['days_since_last_activity'])} dias sin actividad relevante. "
                "Si no se le muestra valor rapido, es probable que nunca se consolide como cliente recurrente."
            ),
        )

    # Clientes con alto riesgo de churn segun el modelo
    if churn_col is not None:
        churn_values = df[churn_col].astype(float)
        unique_values = sorted(churn_values.dropna().unique())
        if unique_values and set(unique_values).issubset({0.0, 1.0}):
            high_risk_mask = churn_values == 1.0
        else:
            high_risk_threshold = 0.7
            high_risk_mask = churn_values >= high_risk_threshold

        add_recommendation(
            mask=high_risk_mask,
            segment="high_churn_risk",
            action="Priorizar contacto proactivo y una oferta de retencion personalizada.",
            reason_builder=lambda row: (
                "El modelo de churn estima una probabilidad alta de abandono para este cliente. "
                "Combinando su comportamiento reciente con esta prediccion, es recomendable "
                "actuar antes de que reduzca aun mas su actividad o se pierda por completo."
            ),
        )

    if not segments:
        logger.info("No se generaron recomendaciones especificas; devolviendo dataframe vacio.")
        return pd.DataFrame(
            columns=[
                "customer_id",
                "segment",
                "recommended_action",
                "reason",
            ]
        )

    recs_df = pd.concat(segments, ignore_index=True)
    logger.info("Recomendaciones generadas para %d clientes", recs_df["customer_id"].nunique())
    return recs_df


def generate_alerts(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un conjunto de alertas de negocio a partir de un dataframe enriquecido
    con características por cliente (features) y, opcionalmente, información de churn.

    Devuelve un dataframe con:
      - segment
      - num_customers
      - description
      - recommended_action
    """
    if analysis_df.empty:
        logger.warning("generate_alerts recibio un dataframe vacio; no se generaran alertas")
        return pd.DataFrame(
            columns=[
                "segment",
                "num_customers",
                "description",
                "recommended_action",
            ]
        )

    df, churn_col, thresholds = _prepare_alert_base_df(analysis_df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "num_customers",
                "description",
                "recommended_action",
            ]
        )

    alerts, _ = _compute_alert_segments(df, churn_col, thresholds)

    if not alerts:
        logger.info("No se encontraron segmentos con condiciones suficientes para generar alertas.")
        return pd.DataFrame(
            [
                {
                    "segment": "no_alerts",
                    "num_customers": 0,
                    "description": "No se detectaron segmentos criticos con los datos actuales.",
                    "recommended_action": (
                        "Seguir monitoreando el comportamiento de los clientes y revisar "
                        "periodicamente las metricas de churn."
                    ),
                }
            ]
        )

    alerts_df = pd.DataFrame(alerts)
    alerts_df = alerts_df.sort_values(by="num_customers", ascending=False).reset_index(drop=True)

    logger.info("Alertas generadas: %d segmentos detectados", len(alerts_df))
    return alerts_df


def get_alert_customers(analysis_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Devuelve, para cada segmento de alerta, el subconjunto de clientes que pertenecen a ese segmento.

    Resultado:
        {
            "inactive_high_value": df_con_clientes_de_ese_segmento,
            "inactive_low_value": df_con_clientes_de_ese_segmento,
            ...
        }
    """
    if analysis_df.empty:
        logger.warning("get_alert_customers recibio un dataframe vacio; no se generaran detalles")
        return {}

    df, churn_col, thresholds = _prepare_alert_base_df(analysis_df)
    if df.empty:
        return {}

    _, segment_masks = _compute_alert_segments(df, churn_col, thresholds)

    segment_customers: Dict[str, pd.DataFrame] = {}
    for segment, mask in segment_masks.items():
        segment_df = df[mask].copy()
        if not segment_df.empty:
            segment_customers[segment] = segment_df

    return segment_customers
