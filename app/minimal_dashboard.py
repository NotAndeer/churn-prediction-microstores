import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistema Churn", layout="wide")
st.title("Sistema Predictivo de Churn")

@st.cache_data
def load_data():
    try:
        features = pd.read_csv('data/processed/features.csv')
        labels = pd.read_csv('data/synthetic/churn_labels.csv')
        return features.merge(labels, on='customer_id')
    except:
        return None

data = load_data()

if data is not None:
    st.metric("Clientes", len(data))
    st.metric("Tasa Churn", f"{data['churn'].mean():.1%}")
    
    fig, ax = plt.subplots()
    data['churn'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)
    
    st.dataframe(data.head())
else:
    st.error("Ejecute: python reproduce.py")