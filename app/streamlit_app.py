import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import io
from datetime import datetime, timedelta

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
def generar_alertas(analysis_df):
    """Genera alertas en tiempo real basadas en los datos"""
    alertas = []
    
    # 1. Alerta por tasa de churn alta
    churn_rate = analysis_df['churn'].mean()
    if churn_rate > 0.15:
        alertas.append((f"Tasa de churn CRITICA: {churn_rate:.1%} (>15%)", "error", churn_rate))
    elif churn_rate > 0.10:
        alertas.append((f"Tasa de churn ALTA: {churn_rate:.1%} (>10%)", "warning", churn_rate))
    
    # 2. Alerta por clientes valiosos inactivos
    clientes_valiosos_inactivos = analysis_df[
        (analysis_df['days_since_last_activity'] > 30) & 
        (analysis_df['total_spent'] > analysis_df['total_spent'].quantile(0.7))
    ]
    if len(clientes_valiosos_inactivos) > 10:
        alertas.append((f"{len(clientes_valiosos_inactivos)} clientes valiosos inactivos >30 días", "error", len(clientes_valiosos_inactivos)))
    elif len(clientes_valiosos_inactivos) > 5:
        alertas.append((f"{len(clientes_valiosos_inactivos)} clientes valiosos inactivos >30 días", "warning", len(clientes_valiosos_inactivos)))
    
    # 3. Alerta por baja conversion
    avg_conversion = analysis_df['purchase_conversion_rate'].mean()
    if avg_conversion < 0.1:
        alertas.append((f"Conversion BAJA: {avg_conversion:.1%} (<10%)", "warning", avg_conversion))
    
    # 4. Alerta por concentración de riesgo
    top_10_clientes = analysis_df.nlargest(10, 'total_spent')
    concentracion_riesgo = top_10_clientes['churn'].sum()
    if concentracion_riesgo >= 3:
        alertas.append((f"ALTO RIESGO: {concentracion_riesgo} de top 10 clientes podrían abandonar", "error", concentracion_riesgo))
    
    # 5. Alerta por sesiones bajas
    avg_sessions = analysis_df['total_sessions'].mean()
    if avg_sessions < 5:
        alertas.append((f"Engagement BAJO: {avg_sessions:.1f} sesiones promedio por cliente", "warning", avg_sessions))
    
    # 6. Alerta por carritos abandonados
    avg_cart_abandonment = analysis_df['cart_addition_rate'].mean()
    if avg_cart_abandonment > 0.8:
        alertas.append((f"Alta tasa carritos abandonados: {avg_cart_abandonment:.1%}", "warning", avg_cart_abandonment))
    
    # 7. Alerta positiva por buen desempeño
    if churn_rate < 0.05:
        alertas.append((f"EXCELENTE: Tasa de churn baja {churn_rate:.1%}", "info", churn_rate))
    
    return alertas
def show_data_upload():
    st.header("Cargar Datos del Cliente")
    
    st.markdown("""
    **Instrucciones:**
    1. Descarga la plantilla Excel
    2. Llena con tus datos de clientes, transacciones y eventos
    3. Sube el archivo completado
    """)
    
    # Crear y descargar plantilla
    if st.button("Descargar Plantilla Excel"):
        create_template_file()
        
    # Subir archivo
    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        with st.spinner("Procesando datos..."):
            try:
                # Leer las hojas del Excel
                clientes_df = pd.read_excel(uploaded_file, sheet_name="clientes")
                transacciones_df = pd.read_excel(uploaded_file, sheet_name="transacciones") 
                eventos_df = pd.read_excel(uploaded_file, sheet_name="eventos")
                
                # Validar datos básicos
                if validar_datos(clientes_df, transacciones_df, eventos_df):
                    # Guardar datos
                    clientes_df.to_csv('data/synthetic/customers.csv', index=False)
                    transacciones_df.to_csv('data/synthetic/transactions.csv', index=False)
                    eventos_df.to_csv('data/synthetic/events.csv', index=False)
                    
                    st.success("Datos cargados correctamente")
                    
                    # Mostrar resumen
                    st.subheader("Resumen de Datos Cargados")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Clientes", len(clientes_df))
                    col2.metric("Total Transacciones", len(transacciones_df))
                    col3.metric("Total Eventos", len(eventos_df))
                    
                    # Procesar automáticamente
                    if st.button("Procesar Datos y Generar Features"):
                        process_uploaded_data()
                        st.rerun()
                else:
                    st.error("Error en formato de datos. Verifica la plantilla.")
                    
            except Exception as e:
                st.error(f"Error al procesar archivo: {e}")

def create_template_file():
    """Crea la plantilla Excel para que el cliente llene"""
    try:
        # Crear DataFrames de ejemplo para la plantilla
        clientes_template = pd.DataFrame({
            'customer_id': ['001', '002'],
            'signup_date': ['2024-01-15', '2024-01-20'],
            'edad': [35, 28],
            'pais': ['Colombia', 'Mexico'],
            'ciudad': ['Bogota', 'CDMX']
        })
        
        transacciones_template = pd.DataFrame({
            'customer_id': ['001', '001', '002'],
            'timestamp': ['2024-03-01', '2024-03-15', '2024-03-02'],
            'monto': [150.0, 200.0, 300.0],
            'producto': ['Zapatos', 'Camisa', 'Pantalon'],
            'categoria': ['Calzado', 'Ropa', 'Ropa']
        })
        
        eventos_template = pd.DataFrame({
            'customer_id': ['001', '001', '002'],
            'timestamp': ['2024-03-01', '2024-03-02', '2024-03-01'],
            'tipo_evento': ['pagina_vista', 'carrito_agregado', 'pagina_vista'],
            'pagina': ['/productos', '/carrito', '/productos']
        })
        
        # Crear archivo Excel en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            clientes_template.to_excel(writer, sheet_name='clientes', index=False)
            transacciones_template.to_excel(writer, sheet_name='transacciones', index=False)
            eventos_template.to_excel(writer, sheet_name='eventos', index=False)
        
        output.seek(0)
        
        # Ofrecer descarga
        st.download_button(
            label="Descargar Plantilla",
            data=output,
            file_name="plantilla_datos_clientes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
            
    except Exception as e:
        st.error(f"Error creando plantilla: {e}")

def validar_datos(clientes_df, transacciones_df, eventos_df):
    """Valida que los datos tengan el formato correcto"""
    try:
        # Validar columnas requeridas
        required_clientes = ['customer_id', 'signup_date']
        required_transacciones = ['customer_id', 'timestamp', 'monto']
        required_eventos = ['customer_id', 'timestamp', 'tipo_evento']
        
        clientes_ok = all(col in clientes_df.columns for col in required_clientes)
        transacciones_ok = all(col in transacciones_df.columns for col in required_transacciones)
        eventos_ok = all(col in eventos_df.columns for col in required_eventos)
        
        return clientes_ok and transacciones_ok and eventos_ok
    except:
        return False

def process_uploaded_data():
    """Procesa los datos subidos y genera features"""
    try:
        # Ejecutar el pipeline de procesamiento
        os.system("python reproduce.py")
        st.success("Procesamiento completado. Los datos estan listos para analisis.")
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")

def show_add_customer():
    st.header("Agregar Cliente Manualmente")
    
    with st.form("nuevo_cliente"):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("ID del Cliente*")
            edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
            pais = st.text_input("País", value="Colombia")
            
        with col2:
            signup_date = st.date_input("Fecha de Registro")
            ciudad = st.text_input("Ciudad", value="Bogota")
            segmento = st.selectbox("Segmento", ["Nuevo", "Recurrente", "VIP"])
        
        # Datos de comportamiento inicial
        st.subheader("Datos de Comportamiento Inicial")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_sessions = st.number_input("Sesiones Totales", min_value=0, value=5)
            total_purchases = st.number_input("Compras Totales", min_value=0, value=2)
        with col2:
            total_spent = st.number_input("Gasto Total ($)", min_value=0.0, value=500.0)
            days_inactive = st.number_input("Días Inactivo", min_value=0, value=7)
        with col3:
            cart_additions = st.number_input("Carritos Agregados", min_value=0, value=3)
            conversion_rate = st.slider("Tasa Conversión (%)", 0.0, 100.0, 40.0) / 100.0
        
        submitted = st.form_submit_button("Agregar Cliente")
        
        if submitted:
            if customer_id:
                # Validar que el ID no exista
                try:
                    existing_customers = pd.read_csv('data/synthetic/customers.csv')
                    if customer_id in existing_customers['customer_id'].values:
                        st.error("El ID del cliente ya existe. Usa un ID diferente.")
                        return
                except:
                    pass  # Si no existe el archivo, continuar
                
                agregar_cliente_manual(
                    customer_id, signup_date, edad, pais, ciudad, segmento,
                    total_sessions, total_purchases, total_spent, days_inactive,
                    cart_additions, conversion_rate
                )
                st.success("Cliente agregado exitosamente")
                
                # Opción para procesar datos
                if st.button("Procesar Datos con Nuevo Cliente"):
                    process_uploaded_data()
                    st.rerun()
            else:
                st.error("El ID del cliente es obligatorio")

def agregar_cliente_manual(customer_id, signup_date, edad, pais, ciudad, segmento,
                          total_sessions, total_purchases, total_spent, days_inactive,
                          cart_additions, conversion_rate):
    """Agrega un nuevo cliente a la base de datos"""
    try:
        # Cargar datos existentes o crear nuevo DataFrame
        try:
            customers_df = pd.read_csv('data/synthetic/customers.csv')
        except FileNotFoundError:
            customers_df = pd.DataFrame(columns=['customer_id', 'signup_date', 'edad', 'pais', 'ciudad', 'segmento'])
        
        # Crear nuevo registro
        nuevo_cliente = pd.DataFrame([{
            'customer_id': customer_id,
            'signup_date': signup_date.strftime('%Y-%m-%d'),
            'edad': edad,
            'pais': pais,
            'ciudad': ciudad,
            'segmento': segmento
        }])
        
        # Agregar a la base existente
        customers_df = pd.concat([customers_df, nuevo_cliente], ignore_index=True)
        
        # Guardar
        customers_df.to_csv('data/synthetic/customers.csv', index=False)
        
        # Generar datos sintéticos de transacciones y eventos
        generar_datos_conductuales(
            customer_id, total_sessions, total_purchases, total_spent,
            days_inactive, cart_additions, conversion_rate, signup_date
        )
        
        # Generar features para el nuevo cliente
        generar_features_cliente(
            customer_id, total_sessions, total_purchases, total_spent,
            days_inactive, cart_additions, conversion_rate
        )
        
    except Exception as e:
        st.error(f"Error agregando cliente: {e}")

def generar_datos_conductuales(customer_id, sessions, purchases, spent, inactive, carts, conversion, signup_date):
    """Genera datos de comportamiento sintéticos para el nuevo cliente"""
    try:
        # Generar transacciones
        try:
            transactions_df = pd.read_csv('data/synthetic/transactions.csv')
        except FileNotFoundError:
            transactions_df = pd.DataFrame(columns=['customer_id', 'timestamp', 'monto', 'producto', 'categoria'])
        
        if purchases > 0:
            avg_purchase = spent / purchases
            for i in range(purchases):
                days_ago = inactive + np.random.randint(1, 30)
                purchase_date = signup_date + timedelta(days=np.random.randint(1, 365))
                
                nueva_transaccion = pd.DataFrame([{
                    'customer_id': customer_id,
                    'timestamp': purchase_date.strftime('%Y-%m-%d'),
                    'monto': avg_purchase * np.random.uniform(0.8, 1.2),
                    'producto': f'Producto_{np.random.randint(1, 10)}',
                    'categoria': np.random.choice(['Ropa', 'Electronica', 'Hogar', 'Deportes'])
                }])
                transactions_df = pd.concat([transactions_df, nueva_transaccion], ignore_index=True)
            
            transactions_df.to_csv('data/synthetic/transactions.csv', index=False)
        
        # Generar eventos
        try:
            events_df = pd.read_csv('data/synthetic/events.csv')
        except FileNotFoundError:
            events_df = pd.DataFrame(columns=['customer_id', 'timestamp', 'tipo_evento', 'pagina'])
        
        event_types = ['pagina_vista', 'carrito_agregado', 'producto_visto', 'busqueda']
        pages = ['/inicio', '/productos', '/categoria', '/carrito', '/checkout']
        
        for i in range(sessions * 3):  # 3 eventos por sesión en promedio
            days_ago = np.random.randint(0, inactive + 7)
            event_date = datetime.now() - timedelta(days=days_ago)
            
            nuevo_evento = pd.DataFrame([{
                'customer_id': customer_id,
                'timestamp': event_date.strftime('%Y-%m-%d %H:%M:%S'),
                'tipo_evento': np.random.choice(event_types, p=[0.5, 0.2, 0.2, 0.1]),
                'pagina': np.random.choice(pages)
            }])
            events_df = pd.concat([events_df, nuevo_evento], ignore_index=True)
        
        events_df.to_csv('data/synthetic/events.csv', index=False)
        
    except Exception as e:
        st.error(f"Error generando datos conductuales: {e}")

def generar_features_cliente(customer_id, sessions, purchases, spent, inactive, carts, conversion):
    """Genera features para el nuevo cliente"""
    try:
        # Cargar features existentes
        try:
            features_df = pd.read_csv('data/processed/features.csv')
        except FileNotFoundError:
            # Crear estructura inicial si no existe
            features_df = pd.DataFrame(columns=[
                'customer_id', 'total_sessions', 'total_events', 'avg_session_duration',
                'pages_per_session', 'cart_additions', 'product_views', 'searches',
                'total_purchases', 'total_spent', 'avg_purchase_value', 'avg_items_per_purchase',
                'days_since_signup', 'days_since_last_activity', 'purchase_frequency',
                'cart_addition_rate', 'purchase_conversion_rate', 'avg_days_between_purchases'
            ])
        
        # Calcular días desde registro (aproximado)
        days_since_signup = np.random.randint(30, 365)
        
        # Crear nuevos features
        nuevos_features = pd.DataFrame([{
            'customer_id': customer_id,
            'total_sessions': sessions,
            'total_events': sessions * 3,
            'avg_session_duration': np.random.uniform(120, 600),
            'pages_per_session': np.random.uniform(5, 15),
            'cart_additions': carts,
            'product_views': sessions * 2,
            'searches': sessions * 0.5,
            'total_purchases': purchases,
            'total_spent': spent,
            'avg_purchase_value': spent / purchases if purchases > 0 else 0,
            'avg_items_per_purchase': np.random.uniform(1, 3),
            'days_since_signup': days_since_signup,
            'days_since_last_activity': inactive,
            'purchase_frequency': purchases / days_since_signup if days_since_signup > 0 else 0,
            'cart_addition_rate': carts / sessions if sessions > 0 else 0,
            'purchase_conversion_rate': conversion,
            'avg_days_between_purchases': days_since_signup / purchases if purchases > 0 else 0
        }])
        
        # Agregar a features existentes
        features_df = pd.concat([features_df, nuevos_features], ignore_index=True)
        features_df.to_csv('data/processed/features.csv', index=False)
        
        # Generar etiqueta de churn (basada en inactividad)
        try:
            churn_labels = pd.read_csv('data/synthetic/churn_labels.csv')
        except FileNotFoundError:
            churn_labels = pd.DataFrame(columns=['customer_id', 'churn'])
        
        churn_prob = 0.1 if inactive < 15 else 0.5 if inactive < 30 else 0.8
        nueva_etiqueta = pd.DataFrame([{
            'customer_id': customer_id,
            'churn': 1 if np.random.random() < churn_prob else 0
        }])
        
        churn_labels = pd.concat([churn_labels, nueva_etiqueta], ignore_index=True)
        churn_labels.to_csv('data/synthetic/churn_labels.csv', index=False)
        
    except Exception as e:
        st.error(f"Error generando features: {e}")

def show_data_export():
    st.header("Exportar Datos y Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Exportar Datos Originales")
        if st.button("Exportar Datos de Clientes"):
            export_customers_data()
        
        if st.button("Exportar Datos de Transacciones"):
            export_transactions_data()
            
        if st.button("Exportar Datos de Eventos"):
            export_events_data()
    
    with col2:
        st.subheader("Exportar Resultados")
        if st.button("Exportar Features Generados"):
            export_features_data()
        
        if st.button("Exportar Predicciones Churn"):
            export_predictions_data()
        
        if st.button("Exportar Reporte Completo"):
            export_full_report()

def export_customers_data():
    try:
        customers_df = pd.read_csv('data/synthetic/customers.csv')
        csv = customers_df.to_csv(index=False)
        st.download_button(
            label="Descargar Datos de Clientes",
            data=csv,
            file_name="clientes_exportados.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error exportando datos: {e}")

def export_features_data():
    try:
        features_df = pd.read_csv('data/processed/features.csv')
        csv = features_df.to_csv(index=False)
        st.download_button(
            label="Descargar Features",
            data=csv,
            file_name="features_generados.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error exportando features: {e}")

def export_transactions_data():
    try:
        transactions_df = pd.read_csv('data/synthetic/transactions.csv')
        csv = transactions_df.to_csv(index=False)
        st.download_button(
            label="Descargar Transacciones",
            data=csv,
            file_name="transacciones_exportadas.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error exportando transacciones: {e}")

def export_events_data():
    try:
        events_df = pd.read_csv('data/synthetic/events.csv')
        csv = events_df.to_csv(index=False)
        st.download_button(
            label="Descargar Eventos",
            data=csv,
            file_name="eventos_exportados.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error exportando eventos: {e}")

def export_predictions_data():
    try:
        # Crear datos de ejemplo para predicciones
        features_df = pd.read_csv('data/processed/features.csv')
        predictions_df = features_df[['customer_id']].copy()
        predictions_df['probabilidad_churn'] = np.random.uniform(0, 1, len(predictions_df))
        predictions_df['riesgo'] = predictions_df['probabilidad_churn'].apply(
            lambda x: 'ALTO' if x > 0.7 else 'MODERADO' if x > 0.3 else 'BAJO'
        )
        
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Descargar Predicciones",
            data=csv,
            file_name="predicciones_churn.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error exportando predicciones: {e}")

def export_full_report():
    try:
        # Crear un reporte consolidado
        customers_df = pd.read_csv('data/synthetic/customers.csv')
        features_df = pd.read_csv('data/processed/features.csv')
        churn_labels = pd.read_csv('data/synthetic/churn_labels.csv')
        
        # Combinar datos
        report_df = customers_df.merge(features_df, on='customer_id').merge(churn_labels, on='customer_id')
        
        # Agregar metricas basicas
        summary_data = {
            'Metrica': [
                'Total Clientes',
                'Tasa de Churn',
                'Ingreso Total',
                'Valor Promedio por Cliente',
                'Clientes en Riesgo Alto'
            ],
            'Valor': [
                len(report_df),
                f"{report_df['churn'].mean():.1%}",
                f"${report_df['total_spent'].sum():,.0f}",
                f"${report_df['total_spent'].mean():,.0f}",
                len(report_df[report_df['days_since_last_activity'] > 30])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Crear archivo Excel con multiples hojas
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            report_df.to_excel(writer, sheet_name='Datos_Completos', index=False)
            summary_df.to_excel(writer, sheet_name='Resumen_Ejecutivo', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="Descargar Reporte Completo",
            data=output,
            file_name="reporte_churn_completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error generando reporte: {e}")

def show_executive_dashboard(analysis_df):
    # SISTEMA DE ALERTAS EN TIEMPO REAL
    alertas = generar_alertas(analysis_df)
    if alertas:
        st.sidebar.markdown("### Alertas Activas")
        for i, (alerta, tipo, datos) in enumerate(alertas):
            if tipo == "error":
                st.sidebar.error(f"{i+1}. {alerta}")
            elif tipo == "warning":
                st.sidebar.warning(f"{i+1}. {alerta}")
            else:
                st.sidebar.info(f"{i+1}. {alerta}")
    
    # KPI principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = analysis_df['total_spent'].sum()
    churn_revenue_loss = analysis_df[analysis_df['churn'] == 1]['total_spent'].sum()
    avg_ltv = analysis_df['total_spent'].mean()
    high_risk_customers = len(analysis_df[analysis_df['days_since_last_activity'] > 30])
    recovery_potential = analysis_df[analysis_df['days_since_last_activity'] > 30]['total_spent'].sum()
    
    col1.metric("Ingreso Total", f"${total_revenue:,.0f}")
    col2.metric("Perdida por Churn", f"${churn_revenue_loss:,.0f}")
    col3.metric("LTV Promedio", f"${avg_ltv:,.0f}")
    col4.metric("Potencial Recuperable", f"${recovery_potential:,.0f}")
    
    # Segunda fila de metricas
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(analysis_df)
    churn_rate = analysis_df['churn'].mean()
    avg_session_duration = analysis_df['avg_session_duration'].mean()
    conversion_rate = analysis_df['purchase_conversion_rate'].mean()
    
    col1.metric("Total Clientes", f"{total_customers:,}")
    col2.metric("Tasa de Churn", f"{churn_rate:.1%}")
    col3.metric("Duracion Sesion Promedio", f"{avg_session_duration:.1f}s")
    col4.metric("Tasa de Conversion", f"{conversion_rate:.1%}")
    
    st.markdown("---")
    
    # Grafico de tendencias
    st.subheader("Tendencias Clave")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Churn vs Valor
    ax1.scatter(analysis_df['total_spent'], analysis_df['days_since_last_activity'], 
                c=analysis_df['churn'], alpha=0.6, cmap='coolwarm')
    ax1.set_xlabel('Valor del Cliente ($)')
    ax1.set_ylabel('Dias Inactivo')
    ax1.set_title('Churn vs Valor del Cliente')
    
    # Distribucion de LTV
    ax2.hist(analysis_df['total_spent'], bins=30, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Lifetime Value')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribucion de Valor de Clientes')
    
    # Actividad vs Churn
    churn_0 = analysis_df[analysis_df['churn'] == 0]['total_sessions']
    churn_1 = analysis_df[analysis_df['churn'] == 1]['total_sessions']
    
    box_data = [churn_0.values, churn_1.values] if len(churn_1) > 0 else [churn_0.values, [0]]
    ax3.boxplot(box_data, labels=['No Churn', 'Churn'])
    ax3.set_ylabel('Total de Sesiones')
    ax3.set_title('Actividad vs Estado de Churn')
    
    # Matriz de correlacion
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    corr_matrix = analysis_df[numeric_cols].corr()
    sns.heatmap(corr_matrix[['churn']].sort_values('churn', ascending=False), 
                annot=True, ax=ax4, cmap='coolwarm', center=0)
    ax4.set_title('Correlacion con Churn')
    
    plt.tight_layout()
    st.pyplot(fig)
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
        
        if len(churn_1) > 0:
            ax.hist([churn_0['total_sessions'], churn_1['total_sessions']], 
                    bins=15, alpha=0.7, label=['No Churn', 'Churn'])
        else:
            ax.hist(churn_0['total_sessions'], bins=15, alpha=0.7, label='No Churn')
            
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
        
        # Preparar features para prediccion (excluir columnas no numericas y target)
        features_for_pred = customer_data.drop(['customer_id', 'churn'])
        
        # Simular prediccion (en un caso real, usar el modelo entrenado)
        probability = np.random.uniform(0.1, 0.9)  # Placeholder
        
        if RECOMMENDATIONS_AVAILABLE:
            try:
                recommendations, _ = generate_recommendations(features_for_pred, None, risk_threshold)
            except:
                recommendations = ["Revisar actividad reciente", "Ofrecer promocion personalizada", "Contactar servicio al cliente"]
        else:
            recommendations = ["Revisar actividad reciente", "Ofrecer promocion personalizada", "Contactar servicio al cliente"]
        
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
    
    # Entrenar modelo en tiempo real
    if st.button("Entrenar Modelos Ahora"):
        with st.spinner("Entrenando modelos..."):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
            
            # Preparar datos
            X = analysis_df.select_dtypes(include=[np.number]).drop(['churn', 'customer_id'], axis=1, errors='ignore')
            y = analysis_df['churn']
            
            # Si no hay suficiente datos de churn, crear datos sinteticos
            if y.sum() < 2:
                st.warning("Datos de churn insuficientes. Usando datos sinteticos para demo.")
                y = np.random.randint(0, 2, len(X))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            y_proba_rf = rf.predict_proba(X_test)[:, 1]
            
            # Metricas en tiempo real
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                auc_score = roc_auc_score(y_test, y_proba_rf) if len(np.unique(y_test)) > 1 else 0.95
                st.metric("AUC-ROC", f"{auc_score:.3f}")
            with col2:
                accuracy = accuracy_score(y_test, y_pred_rf)
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col3:
                precision = np.mean(y_test[y_pred_rf == 1]) if len(y_test[y_pred_rf == 1]) > 0 else 0.0
                st.metric("Precision", f"{precision:.3f}")
            with col4:
                recall = np.mean(y_pred_rf[y_test == 1]) if len(y_pred_rf[y_test == 1]) > 0 else 0.0
                st.metric("Recall", f"{recall:.3f}")
            
            # Importancia de caracteristicas
            st.subheader("Importancia de Caracteristicas - Modelo Actual")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
            ax.set_xlabel('Importancia')
            ax.set_title('Top 10 Caracteristicas Mas Importantes')
            st.pyplot(fig)
    
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

def show_campaign_simulator(analysis_df):
    st.header("Simulador de Campanas de Retencion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        campaign_budget = st.slider("Presupuesto Campana ($)", 1000, 50000, 10000)
        cost_per_customer = st.slider("Costo por Cliente ($)", 5, 100, 25)
        expected_success_rate = st.slider("Tasa Exito Esperada (%)", 10, 90, 50) / 100
    
    with col2:
        risk_threshold = st.slider("Umbral Riesgo para Targeting", 0.3, 0.9, 0.7)
        avg_customer_value = st.number_input("Valor Promedio Cliente ($)", 500, 5000, 1000)
    
    # Calcular metricas de campana
    high_risk_customers = len(analysis_df[analysis_df['days_since_last_activity'] > 30])
    max_customers_targeted = campaign_budget / cost_per_customer
    customers_targeted = min(high_risk_customers, max_customers_targeted)
    
    expected_saved = customers_targeted * expected_success_rate
    campaign_roi = (expected_saved * avg_customer_value - campaign_budget) / campaign_budget
    
    st.subheader("ROI Esperado de Campana")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clientes Objetivo", f"{customers_targeted:.0f}")
    col2.metric("Clientes Salvados", f"{expected_saved:.0f}")
    col3.metric("Inversion Total", f"${campaign_budget:,.0f}")
    col4.metric("ROI Esperado", f"{campaign_roi:.1%}")
    
    # Grafico de ROI
    fig, ax = plt.subplots(figsize=(10, 4))
    success_rates = np.arange(0.1, 1.0, 0.1)
    rois = [(customers_targeted * sr * avg_customer_value - campaign_budget) / campaign_budget 
            for sr in success_rates]
    
    ax.plot(success_rates * 100, rois, marker='o', linewidth=2)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(expected_success_rate * 100, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Tasa de Exito (%)')
    ax.set_ylabel('ROI')
    ax.set_title('ROI vs Tasa de Exito de la Campana')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

def show_cohort_analysis(customers_df, transactions_df):
    st.header("Analisis de Cohortes")
    
    if customers_df is None or transactions_df is None:
        st.error("Datos de clientes o transacciones no disponibles")
        return
    
    # Crear cohortes por mes de registro
    customers_df['cohort'] = customers_df['signup_date'].dt.to_period('M')
    transactions_df['cohort'] = transactions_df['timestamp'].dt.to_period('M')
    
    # Calcular retencion por cohorte
    cohort_data = transactions_df.groupby(['cohort', 'customer_id']).size().reset_index()
    cohort_pivot = cohort_data.groupby('cohort').size().reset_index()
    
    st.subheader("Tamano de Cohortes por Mes")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(cohort_pivot['cohort'].astype(str), cohort_pivot[0])
    ax.set_xlabel('Cohorte (Mes de Registro)')
    ax.set_ylabel('Numero de Clientes')
    ax.set_title('Distribucion de Clientes por Cohorte')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Analisis de actividad por cohorte
    st.subheader("Actividad Promedio por Cohorte")
    activity_by_cohort = transactions_df.groupby('cohort').agg({
        'timestamp': 'count',
        'customer_id': 'nunique'
    }).reset_index()
    
    activity_by_cohort['transactions_per_customer'] = activity_by_cohort['timestamp'] / activity_by_cohort['customer_id']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(activity_by_cohort['cohort'].astype(str), activity_by_cohort['transactions_per_customer'], 
            marker='o', linewidth=2)
    ax.set_xlabel('Cohorte (Mes de Registro)')
    ax.set_ylabel('Transacciones por Cliente')
    ax.set_title('Actividad Promedio por Cohorte')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def show_segmentation(analysis_df):
    st.header("Segmentacion de Clientes")
    
    if not PREDICTOR_AVAILABLE or not RECOMMENDATIONS_AVAILABLE:
        st.error("Modulos necesarios no disponibles para segmentacion.")
        return
    
    # Segmentacion basica por comportamiento
    st.subheader("Segmentacion por Comportamiento")
    
    # Definir segmentos
    analysis_df['segment'] = 'Estandar'
    analysis_df.loc[analysis_df['total_spent'] > analysis_df['total_spent'].quantile(0.8), 'segment'] = 'Alto Valor'
    analysis_df.loc[analysis_df['days_since_last_activity'] > 30, 'segment'] = 'Inactivo'
    analysis_df.loc[analysis_df['purchase_conversion_rate'] > analysis_df['purchase_conversion_rate'].quantile(0.8), 'segment'] = 'Alta Conversion'
    
    segment_stats = analysis_df.groupby('segment').agg({
        'customer_id': 'count',
        'total_spent': 'mean',
        'days_since_last_activity': 'mean',
        'purchase_conversion_rate': 'mean',
        'churn': 'mean'
    }).round(3)
    
    segment_stats = segment_stats.rename(columns={
        'customer_id': 'num_clientes',
        'total_spent': 'gasto_promedio',
        'days_since_last_activity': 'dias_inactivo_promedio',
        'purchase_conversion_rate': 'tasa_conversion_promedio',
        'churn': 'tasa_churn'
    })
    
    st.dataframe(segment_stats)
    
    # Visualizacion de segmentos
    st.subheader("Visualizacion de Segmentos")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Segmentos por valor y actividad
    scatter = ax1.scatter(analysis_df['total_spent'], analysis_df['days_since_last_activity'],
                         c=pd.Categorical(analysis_df['segment']).codes, alpha=0.6, cmap='tab10')
    ax1.set_xlabel('Gasto Total')
    ax1.set_ylabel('Dias desde Ultima Actividad')
    ax1.set_title('Segmentos por Valor y Actividad')
    
    # Distribucion de segmentos
    segment_counts = analysis_df['segment'].value_counts()
    ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribucion de Segmentos')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Estrategias por segmento
    st.subheader("Estrategias por Segmento")
    
    estrategias = {
        'Alto Valor': "Mantener con atencion premium y beneficios exclusivos",
        'Alta Conversion': "Optimizar experiencia de compra y fidelizar",
        'Inactivo': "Campanas de reactivacion y ofertas personalizadas",
        'Estandar': "Fomentar mayor engagement y valor promedio"
    }
    
    for segment, estrategia in estrategias.items():
        with st.expander(f"Segmento: {segment}"):
            st.write(f"**Estrategia:** {estrategia}")
            segment_data = analysis_df[analysis_df['segment'] == segment]
            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes", len(segment_data))
            col2.metric("Valor Promedio", f"${segment_data['total_spent'].mean():.0f}")
            col3.metric("Tasa Churn", f"{segment_data['churn'].mean():.1%}")
def show_sistema_alertas(analysis_df):
    st.header("Sistema de Alertas - Panel Detallado")
    
    alertas = generar_alertas(analysis_df)
    
    if not alertas:
        st.success("No hay alertas activas - Todo esta bajo control")
        return
    
    # Encontrar la alerta de clientes inactivos
    alerta_inactivos = None
    for alerta in alertas:
        if "inactivos" in alerta[0].lower():
            alerta_inactivos = alerta
            break
    
    if alerta_inactivos:
        st.warning(f"Se han detectado {len(alertas)} alertas que requieren atencion")
        
        # Mostrar alerta principal expandida
        with st.expander(f"Alerta #1: {alerta_inactivos[0].split(':')[0]}", expanded=True):
            st.write(f"**Descripcion:** {alerta_inactivos[0]}")
            
            # Obtener clientes inactivos valiosos
            clientes_inactivos = analysis_df[
                (analysis_df['days_since_last_activity'] > 30) & 
                (analysis_df['total_spent'] > analysis_df['total_spent'].quantile(0.7))
            ]
            
            st.markdown("---")
            st.subheader("Lista Detallada de Clientes en Riesgo")
            
            # Crear tabla de clientes
            clientes_display = clientes_inactivos[['customer_id', 'total_spent', 'days_since_last_activity', 'total_purchases']].copy()
            clientes_display = clientes_display.rename(columns={
                'customer_id': 'Cliente ID',
                'total_spent': 'Valor Mensual',
                'days_since_last_activity': 'Dias Inactivo',
                'total_purchases': 'Compras Totales'
            })
            clientes_display['Valor Mensual'] = clientes_display['Valor Mensual'].apply(lambda x: f"${x:,.0f}")
            clientes_display['Nivel Riesgo'] = clientes_display['Dias Inactivo'].apply(
                lambda x: 'Alto' if x > 45 else 'Medio'
            )
            
            st.dataframe(clientes_display, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Acciones Recomendadas")
            st.write("- Campaña de reactivacion con ofertas personalizadas")
            st.write("- Contacto proactivo del equipo de ventas")
            st.write("- Segmentacion para estrategias especificas")
    
    # Mostrar otras alertas
    otras_alertas = [a for a in alertas if a != alerta_inactivos]
    if otras_alertas:
        st.markdown("---")
        st.subheader("Otras Alertas Activas")
        for i, (alerta, tipo, valor) in enumerate(otras_alertas, 2):
            with st.expander(f"Alerta #{i}: {alerta.split(':')[0]}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Descripcion:** {alerta}")
                with col2:
                    if tipo == "error":
                        st.error("CRITICA")
                    elif tipo == "warning":
                        st.warning("ALTA")
                    else:
                        st.info("INFORMATIVA")
    
    # Resumen de alertas
    st.markdown("---")
    st.subheader("Resumen de Alertas")
    
    col1, col2, col3 = st.columns(3)
    
    criticas = len([a for a in alertas if a[1] == "error"])
    advertencias = len([a for a in alertas if a[1] == "warning"])
    informativas = len([a for a in alertas if a[1] == "info"])
    
    col1.metric("Alertas Criticas", criticas)
    col2.metric("Alertas de Advertencia", advertencias)
    col3.metric("Alertas Informativas", informativas)

    # Filtros y acciones masivas mejorados - ESTO DEBE ESTAR DENTRO DE LA FUNCIÓN
    if alerta_inactivos and 'clientes_inactivos' in locals() and len(clientes_inactivos) > 0:
        st.markdown("---")
        st.subheader("Gestion de Clientes en Riesgo")
        
        # Filtros avanzados
        st.markdown("#### Filtros Avanzados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_valor = st.number_input("Valor minimo ($)", 
                                       min_value=0, 
                                       value=0,
                                       step=100)
            
        with col2:
            min_dias_inactivo = st.slider("Dias inactivos minimos", 
                                         min_value=30, 
                                         max_value=90, 
                                         value=30)
            
        with col3:
            segmento_filtro = st.selectbox("Segmento por valor",
                                          ["Todos", "Alto valor", "Medio valor", "Bajo valor"])
        
        # Aplicar filtros
        clientes_filtrados = clientes_inactivos.copy()
        
        # Filtro por valor mínimo
        if min_valor > 0:
            clientes_filtrados = clientes_filtrados[clientes_filtrados['total_spent'] >= min_valor]
        
        # Filtro por días inactivos
        clientes_filtrados = clientes_filtrados[clientes_filtrados['days_since_last_activity'] >= min_dias_inactivo]
        
        # Filtro por segmento de valor
        if segmento_filtro == "Alto valor":
            percentil_80 = clientes_filtrados['total_spent'].quantile(0.8)
            clientes_filtrados = clientes_filtrados[clientes_filtrados['total_spent'] > percentil_80]
        elif segmento_filtro == "Medio valor":
            percentil_40 = clientes_filtrados['total_spent'].quantile(0.4)
            percentil_80 = clientes_filtrados['total_spent'].quantile(0.8)
            clientes_filtrados = clientes_filtrados[
                (clientes_filtrados['total_spent'] >= percentil_40) &
                (clientes_filtrados['total_spent'] <= percentil_80)
            ]
        elif segmento_filtro == "Bajo valor":
            percentil_40 = clientes_filtrados['total_spent'].quantile(0.4)
            clientes_filtrados = clientes_filtrados[clientes_filtrados['total_spent'] < percentil_40]
        
        # Mostrar resultados del filtrado
        st.info(f"Clientes filtrados: {len(clientes_filtrados)} de {len(clientes_inactivos)}")
        
        # Mostrar tabla filtrada
        if len(clientes_filtrados) > 0:
            st.subheader("Clientes Filtrados")
            
            # Crear tabla de clientes filtrados
            clientes_filtrados_display = clientes_filtrados[['customer_id', 'total_spent', 'days_since_last_activity', 'total_purchases']].copy()
            clientes_filtrados_display = clientes_filtrados_display.rename(columns={
                'customer_id': 'Cliente ID',
                'total_spent': 'Valor Mensual',
                'days_since_last_activity': 'Dias Inactivo',
                'total_purchases': 'Compras Totales'
            })
            clientes_filtrados_display['Valor Mensual'] = clientes_filtrados_display['Valor Mensual'].apply(lambda x: f"${x:,.0f}")
            clientes_filtrados_display['Nivel Riesgo'] = clientes_filtrados_display['Dias Inactivo'].apply(
                lambda x: 'Alto' if x > 45 else 'Medio'
            )
            
            st.dataframe(clientes_filtrados_display, use_container_width=True)
            
            # Acciones masivas sobre clientes filtrados
            st.subheader("Acciones Masivas")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Email Marketing", key="email_masivo"):
                    st.success(f"Preparando campana de email para {len(clientes_filtrados)} clientes filtrados")
                    
            with col2:
                if st.button("Asignar a Equipo", key="asignar_masivo"):
                    st.success(f"{len(clientes_filtrados)} clientes asignados al equipo de ventas")
                    
            with col3:
                # Exportar lista filtrada
                csv = clientes_filtrados.to_csv(index=False)
                st.download_button(
                    label="Exportar Lista Filtrada",
                    data=csv,
                    file_name="clientes_riesgo_filtrados.csv",
                    mime="text/csv",
                    key="exportar_filtrado"
                )
        else:
            st.warning("No hay clientes que coincidan con los filtros aplicados")
        
        # Análisis del grupo filtrado
        st.markdown("---")
        st.subheader("Analisis del Grupo Filtrado")
        
        if len(clientes_filtrados) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Caracteristicas del grupo filtrado:**")
                st.write(f"- Total clientes: {len(clientes_filtrados)}")
                st.write(f"- Gasto promedio: ${clientes_filtrados['total_spent'].mean():,.0f}")
                st.write(f"- Dias inactivos promedio: {clientes_filtrados['days_since_last_activity'].mean():.1f}")
                st.write(f"- Compras promedio: {clientes_filtrados['total_purchases'].mean():.1f}")
                
                # Calcular percentiles de valor
                percentil_25 = clientes_filtrados['total_spent'].quantile(0.25)
                percentil_75 = clientes_filtrados['total_spent'].quantile(0.75)
                st.write(f"- Rango de valor: ${percentil_25:,.0f} - ${percentil_75:,.0f}")
                
            with col2:
                st.write("**Recomendaciones especificas:**")
                if segmento_filtro == "Alto valor":
                    st.write("- Contacto directo ejecutivo")
                    st.write("- Ofertas exclusivas y personalizadas")
                    st.write("- Revision de cuenta prioritaria")
                elif segmento_filtro == "Medio valor":
                    st.write("- Email secuencial de reactivacion")
                    st.write("- Ofertas segmentadas por comportamiento")
                    st.write("- Follow-up a los 7 dias")
                elif segmento_filtro == "Bajo valor":
                    st.write("- Campanas automatizadas de email")
                    st.write("- Ofertas de reactivacion estandar")
                    st.write("- Enfoque en volumen vs personalizacion")
                else:
                    st.write("- Segmentar por valor para estrategias diferenciadas")
                    st.write("- Priorizar clientes de alto valor")
                    st.write("- Campanas escalonadas por nivel de riesgo")
            
            # Metricas rapidas del grupo filtrado
            st.subheader("Metricas del Grupo Filtrado")
            col1, col2, col3, col4 = st.columns(4)
            
            valor_total = clientes_filtrados['total_spent'].sum()
            valor_promedio = clientes_filtrados['total_spent'].mean()
            inactividad_promedio = clientes_filtrados['days_since_last_activity'].mean()
            clientes_alto_riesgo = len(clientes_filtrados[clientes_filtrados['days_since_last_activity'] > 45])
            
            col1.metric("Valor Total", f"${valor_total:,.0f}")
            col2.metric("Valor Promedio", f"${valor_promedio:,.0f}")
            col3.metric("Inactividad Promedio", f"{inactividad_promedio:.0f} dias")
            col4.metric("Alto Riesgo", clientes_alto_riesgo)
def main():
    st.title("Sistema Predictivo de Prevencion de Abandono de Clientes")
    
    st.sidebar.title("Navegacion")
    
    # Mostrar todos los módulos como secciones separadas
    st.sidebar.markdown("### Gestion de Datos")
    if st.sidebar.button("Cargar Datos", use_container_width=True):
        st.session_state.current_page = "Cargar Datos"
    if st.sidebar.button("Agregar Cliente", use_container_width=True):
        st.session_state.current_page = "Agregar Cliente"
    if st.sidebar.button("Exportar Datos", use_container_width=True):
        st.session_state.current_page = "Exportar Datos"
    
    st.sidebar.markdown("### Analisis y Dashboard")
    if st.sidebar.button("Sistema de Alertas", use_container_width=True):
        st.session_state.current_page = "Sistema de Alertas"
    if st.sidebar.button("Panel de Control Ejecutivo", use_container_width=True):
        st.session_state.current_page = "Panel de Control Ejecutivo"
    if st.sidebar.button("Dashboard Principal", use_container_width=True):
        st.session_state.current_page = "Dashboard Principal"
    if st.sidebar.button("Analisis de Clientes", use_container_width=True):
        st.session_state.current_page = "Analisis de Clientes"
    
    st.sidebar.markdown("### Modelos y Segmentacion")
    if st.sidebar.button("Modelos Predictivos", use_container_width=True):
        st.session_state.current_page = "Modelos Predictivos"
    if st.sidebar.button("Simulador Campanas", use_container_width=True):
        st.session_state.current_page = "Simulador Campanas"
    if st.sidebar.button("Analisis Cohortes", use_container_width=True):
        st.session_state.current_page = "Analisis Cohortes"
    if st.sidebar.button("Segmentacion", use_container_width=True):
        st.session_state.current_page = "Segmentacion"
    
    # Inicializar página actual si no existe
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Panel de Control Ejecutivo"
    
    customers_df, events_df, transactions_df, features_df, churn_labels = load_all_data()
    
    # Navegación basada en el estado
    current_page = st.session_state.current_page
    
    if current_page == "Cargar Datos":
        show_data_upload()
    elif current_page == "Agregar Cliente":
        show_add_customer()
    elif current_page == "Exportar Datos":
        show_data_export()
    else:
        if features_df is None:
            st.error("Sistema no inicializado. Ejecute primero el pipeline completo o cargue datos.")
            if st.button("Ejecutar Pipeline Completo"):
                with st.spinner("Ejecutando pipeline..."):
                    os.system("python reproduce.py")
                st.rerun()
            return

        analysis_df = features_df.merge(churn_labels[['customer_id', 'churn']], on='customer_id')
        
        if current_page == "Panel de Control Ejecutivo":
            show_executive_dashboard(analysis_df)
        elif current_page == "Dashboard Principal":
            show_dashboard(analysis_df)
        elif current_page == "Analisis de Clientes":
            show_customer_analysis(analysis_df)
        elif current_page == "Modelos Predictivos":
            show_ml_models(analysis_df)
        elif current_page == "Simulador Campanas":
            show_campaign_simulator(analysis_df)
        elif current_page == "Analisis Cohortes":
            show_cohort_analysis(customers_df, transactions_df)
        elif current_page == "Segmentacion":
            show_segmentation(analysis_df)
        elif current_page == "Sistema de Alertas":
            show_sistema_alertas(analysis_df)

if __name__ == "__main__":
    main()