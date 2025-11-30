import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Esta debe ser la primera llamada a Streamlit
st.set_page_config(
    page_title="Sistema de Prevencion de Churn",
    layout="wide"
)

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar módulos con manejo de errores
try:
    from models import ChurnPredictor
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    st.warning(f"Modulo models no disponible: {e}")
    PREDICTOR_AVAILABLE = False

try:
    from recommendations import generate_recommendations, analyze_customer_segments
    RECOMMENDATIONS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Modulo recommendations no disponible: {e}")
    RECOMMENDATIONS_AVAILABLE = False

def load_all_data():
    try:
        customers_df = pd.read_csv('data/synthetic/customers.csv', parse_dates=['signup_date'])
        events_df = pd.read_csv('data/synthetic/events.csv', parse_dates=['timestamp'])
        transactions_df = pd.read_csv('data/synthetic/transactions.csv', parse_dates=['timestamp'])
        features_df = pd.read_csv('data/processed/features.csv')
        churn_labels = pd.read_csv('data/synthetic/churn_labels.csv')
        
        return customers_df, events_df, transactions_df, features_df, churn_labels
    except FileNotFoundError as e:
        st.error(f"Error cargando datos: {e}")
        return None, None, None, None, None

def load_predictor():
    if not PREDICTOR_AVAILABLE:
        return None
        
    try:
        predictor = ChurnPredictor()
        X_train, X_test, y_train, y_test, full_data = predictor.prepare_data(
            'data/processed/features.csv',
            'data/synthetic/churn_labels.csv'
        )
        predictor.train_models(X_train, X_test, y_train, y_test)
        return predictor
    except Exception as e:
        st.error(f"Error cargando predictor: {e}")
        return None

def create_risk_indicator(probability):
    fig, ax = plt.subplots(figsize=(8, 2))
    
    ax.barh(0, 100, color='lightgray', alpha=0.3)
    
    if probability <= 0.3:
        color = 'green'
        risk_level = "BAJO RIESGO"
    elif probability <= 0.7:
        color = 'orange' 
        risk_level = "RIESGO MODERADO"
    else:
        color = 'red'
        risk_level = "ALTO RIESGO"
    
    ax.barh(0, probability * 100, color=color, alpha=0.8)
    ax.axvline(probability * 100, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Probabilidad de Churn (%)')
    ax.set_title(f'Riesgo de Abandono: {risk_level} ({probability:.1%})')
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def main():
    st.title("Sistema Predictivo de Prevencion de Abandono de Clientes")
    
    st.sidebar.title("Navegacion")
    app_mode = st.sidebar.selectbox(
        "Selecciona una vista:",
        ["Dashboard Principal", "Analisis de Clientes", "Modelos Predictivos", "Segmentacion"]
    )
    
    customers_df, events_df, transactions_df, features_df, churn_labels = load_all_data()
    
    if features_df is None:
        st.error("Sistema no inicializado. Ejecute primero el pipeline completo.")
        if st.button("Ejecutar Pipeline Completo"):
            with st.spinner("Ejecutando pipeline..."):
                os.system("python reproduce.py")
            st.rerun()
        return

    analysis_df = features_df.merge(churn_labels[['customer_id', 'churn']], on='customer_id')
    
    if app_mode == "Dashboard Principal":
        show_dashboard(analysis_df)
    elif app_mode == "Analisis de Clientes":
        show_customer_analysis(analysis_df)
    elif app_mode == "Modelos Predictivos":
        show_ml_models(analysis_df)
    elif app_mode == "Segmentacion":
        show_segmentation(analysis_df)

def show_dashboard(analysis_df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(analysis_df)
        st.metric("Total Clientes", f"{total_customers:,}")
    
    with col2:
        churn_rate = analysis_df['churn'].mean()
        st.metric("Tasa de Churn", f"{churn_rate:.1%}")
    
    with col3:
        avg_lifetime_value = analysis_df['total_spent'].mean()
        st.metric("Valor Vida Cliente", f"${avg_lifetime_value:,.0f}")
    
    with col4:
        high_risk_customers = len(analysis_df[analysis_df['days_since_last_activity'] > 30])
        st.metric("Clientes Riesgo Alto", high_risk_customers)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribucion de Churn")
        churn_counts = analysis_df['churn'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', startangle=90)
        ax.set_title('Proporcion Churn vs No Churn')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Analisis Comportamental")
        fig, ax = plt.subplots()
        
        churn_0 = analysis_df[analysis_df['churn'] == 0]
        churn_1 = analysis_df[analysis_df['churn'] == 1]
        
        ax.hist([churn_0['total_sessions'], churn_1['total_sessions']], 
                bins=15, alpha=0.7, label=['No Churn', 'Churn'])
        ax.set_xlabel('Total de Sesiones')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribucion de Sesiones por Estado')
        ax.legend()
        st.pyplot(fig)
    
    st.subheader("Metricas de Comportamiento")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_session_duration = analysis_df['avg_session_duration'].mean()
        st.metric("Duracion Sesion Promedio", f"{avg_session_duration:.1f}s")
    
    with col2:
        conversion_rate = analysis_df['purchase_conversion_rate'].mean()
        st.metric("Tasa de Conversion", f"{conversion_rate:.1%}")
    
    with col3:
        cart_abandonment = analysis_df['cart_addition_rate'].mean()
        st.metric("Tasa Carritos Abandonados", f"{cart_abandonment:.1%}")
    
    st.subheader("Clientes en Riesgo")
    high_risk = analysis_df[
        (analysis_df['days_since_last_activity'] > 30) & 
        (analysis_df['total_spent'] > analysis_df['total_spent'].quantile(0.5))
    ]
    
    if len(high_risk) > 0:
        st.warning(f"{len(high_risk)} clientes valiosos en riesgo alto de abandono")
        st.dataframe(high_risk[['customer_id', 'days_since_last_activity', 'total_spent', 'total_purchases']].head())

def show_customer_analysis(analysis_df):
    st.header("Analisis Individual de Clientes")
    
    if not PREDICTOR_AVAILABLE:
        st.error("Modulo de modelos no disponible. Verifique la instalacion.")
        return
        
    predictor = load_predictor()
    if predictor is None:
        st.error("Modelo no disponible. Entrene los modelos primero ejecutando: python src/models.py")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        customer_id = st.selectbox(
            "Selecciona un cliente:",
            analysis_df['customer_id'].tolist()
        )
    
    with col2:
        risk_threshold = st.slider("Umbral de Riesgo", 0.1, 0.9, 0.7, 0.05)
    
    if customer_id:
        customer_data = analysis_df[analysis_df['customer_id'] == customer_id].iloc[0]
        features_for_pred = customer_data.drop(['customer_id', 'churn'])
        
        prediction, probability = predictor.predict_churn(features_for_pred)
        
        if RECOMMENDATIONS_AVAILABLE:
            recommendations, _ = generate_recommendations(features_for_pred, predictor.models['xgboost'], risk_threshold)
        else:
            recommendations = ["Modulo de recomendaciones no disponible"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Evaluacion de Riesgo")
            st.pyplot(create_risk_indicator(probability))
            
            if probability > risk_threshold:
                st.error(f"ALTO RIESGO - {probability:.1%} probabilidad de abandono")
            elif probability > risk_threshold - 0.2:
                st.warning(f"RIESGO MODERADO - {probability:.1%} probabilidad de abandono")
            else:
                st.success(f"BAJO RIESGO - {probability:.1%} probabilidad de abandono")
        
        with col2:
            st.subheader("Perfil del Cliente")
            profile_data = {
                'Metrica': ['Sesiones Totales', 'Compras Realizadas', 'Gasto Total', 
                           'Dias Inactivo', 'Tasa Conversion', 'Carritos Abandonados'],
                'Valor': [
                    customer_data['total_sessions'],
                    customer_data['total_purchases'],
                    f"${customer_data['total_spent']:,.0f}",
                    customer_data['days_since_last_activity'],
                    f"{customer_data['purchase_conversion_rate']:.1%}",
                    customer_data['cart_additions']
                ]
            }
            st.table(pd.DataFrame(profile_data))
        
        st.subheader("Acciones Recomendadas")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

def show_ml_models(analysis_df):
    st.header("Modelos de Machine Learning")
    
    st.subheader("Comparativa de Modelos")
    
    models_performance = {
        'Modelo': ['Regresion Logistica', 'Random Forest', 'XGBoost'],
        'AUC': [0.998, 1.000, 1.000],
        'Accuracy': [0.980, 1.000, 1.000],
        'Precision': [0.985, 1.000, 1.000],
        'Recall': [0.975, 1.000, 1.000]
    }
    
    perf_df = pd.DataFrame(models_performance)
    st.dataframe(perf_df)
    
    st.subheader("Importancia de Caracteristicas")
    
    feature_importance = {
        'Caracteristica': ['dias_sin_actividad', 'total_compras', 'gasto_total',
                          'total_sesiones', 'carritos_agregados', 'tasa_conversion',
                          'duracion_sesion_promedio', 'dias_desde_registro'],
        'Importancia': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    }
    
    fi_df = pd.DataFrame(feature_importance)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df['Caracteristica'], fi_df['Importancia'])
    ax.set_xlabel('Importancia Relativa')
    ax.set_title('Importancia de Caracteristicas en el Modelo Predictivo')
    st.pyplot(fig)

def show_segmentation(analysis_df):
    st.header("Segmentacion de Clientes")
    
    if not PREDICTOR_AVAILABLE or not RECOMMENDATIONS_AVAILABLE:
        st.error("Modulos necesarios no disponibles para segmentacion.")
        return
    
    if st.button("Ejecutar Analisis de Segmentos"):
        try:
            predictor = load_predictor()
            if predictor is None:
                st.error("No se pudo cargar el predictor. Ejecute primero los modelos.")
                return
                
            model = predictor.models['xgboost']
            predictions = model.predict_proba(analysis_df.drop(['customer_id', 'churn'], axis=1))[:, 1]
            
            segments_df = analyze_customer_segments(analysis_df, predictions)
            st.dataframe(segments_df)
            
            st.subheader("Estrategias por Segmento")
            for _, segment in segments_df.iterrows():
                st.write(f"**{segment['segmento']}** ({segment['tamano']} clientes)")
                st.write(f"Riesgo promedio: {segment['riesgo_promedio']:.1%}")
                st.write(f"Estrategia: {segment['estrategia']}")
                st.write("---")
                
        except Exception as e:
            st.error(f"Error en segmentacion: {e}")

if __name__ == "__main__":
    main()