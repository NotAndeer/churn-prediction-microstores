import pandas as pd
import numpy as np

def generate_recommendations(customer_data, model, risk_threshold=0.7):
    """
    Genera recomendaciones basadas en los datos del cliente y prediccion del modelo
    """
    
    probability = model.predict_proba(customer_data.values.reshape(1, -1))[0, 1]
    
    if probability < risk_threshold:
        return ["Cliente de bajo riesgo - Mantener engagement natural"], probability
    
    recommendations = []
    
    # Analizar caracteristicas especificas para recomendaciones
    if 'days_since_last_activity' in customer_data.index:
        days_inactive = customer_data['days_since_last_activity']
        if days_inactive > 30:
            recommendations.append(
                f"Cliente inactivo por {int(days_inactive)} dias. Enviar email de reactivacion con cupon del 15%"
            )
    
    if 'cart_additions' in customer_data.index:
        carts = customer_data['cart_additions']
        if carts > 0:
            recommendations.append(
                f"Tiene {int(carts)} productos en carrito. Ofrecer envio gratis en esos productos"
            )
    
    if 'total_purchases' in customer_data.index:
        purchases = customer_data['total_purchases']
        if purchases == 0:
            recommendations.append(
                "Cliente nunca ha comprado. Ofrecer descuento de primer compra"
            )
        elif purchases > 3:
            recommendations.append(
                f"Cliente frecuente ({int(purchases)} compras). Ofrecer programa de fidelizacion"
            )
    
    if 'avg_session_duration' in customer_data.index:
        session_duration = customer_data['avg_session_duration']
        if session_duration < 60:
            recommendations.append(
                f"Sesiones muy cortas ({session_duration:.1f} segundos). Mejorar experiencia de usuario"
            )
    
    if not recommendations:
        recommendations.append("Revisar manualmente - Caso complejo")
    
    return recommendations, probability

def analyze_customer_segments(features_df, churn_predictions):
    """
    Analiza segmentos de clientes para estrategias de retencion
    """
    data = features_df.copy()
    data['churn_probability'] = churn_predictions
    
    segments = []
    
    # Segmento: Clientes inactivos
    inactive_customers = data[data['days_since_last_activity'] > 45]
    if len(inactive_customers) > 0:
        segments.append({
            'segmento': 'Clientes Inactivos',
            'tamano': len(inactive_customers),
            'riesgo_promedio': inactive_customers['churn_probability'].mean(),
            'estrategia': 'Campaña de reactivacion con incentivos'
        })
    
    # Segmento: Carritos abandonados
    cart_abandoners = data[data['cart_additions'] > 0]
    if len(cart_abandoners) > 0:
        segments.append({
            'segmento': 'Abandonan Carritos',
            'tamano': len(cart_abandoners),
            'riesgo_promedio': cart_abandoners['churn_probability'].mean(),
            'estrategia': 'Recordatorios y ofertas en productos del carrito'
        })
    
    # Segmento: Compradores frecuentes en riesgo
    frequent_buyers = data[data['total_purchases'] > 2]
    high_risk_frequent = frequent_buyers[frequent_buyers['churn_probability'] > 0.6]
    if len(high_risk_frequent) > 0:
        segments.append({
            'segmento': 'Compradores Frecuentes en Riesgo',
            'tamano': len(high_risk_frequent),
            'riesgo_promedio': high_risk_frequent['churn_probability'].mean(),
            'estrategia': 'Programa de fidelizacion premium'
        })
    
    return pd.DataFrame(segments)

if __name__ == "__main__":
    print("Modulo de recomendaciones cargado correctamente")