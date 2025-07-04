import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Configuración general
st.set_page_config(page_title="Predicción de Riesgo de Diabetes", layout="wide")
st.title("🩺 Predicción de Riesgo de Diabetes")
st.markdown("Esta aplicación permite analizar factores clínicos y de estilo de vida para estimar el riesgo de desarrollar diabetes.")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_entrenado.pkl")

model = load_model()

# Sidebar
st.sidebar.header("🧾 Ingreso de Datos del Paciente")

def user_input_features():
    age = st.sidebar.slider("Edad", 18, 100, 45)
    bmi = st.sidebar.slider("Índice de Masa Corporal (BMI)", 15.0, 45.0, 25.0)
    glucose = st.sidebar.slider("Glucosa en Ayunas (mg/dL)", 70, 200, 100)
    physical = st.sidebar.selectbox("Nivel de Actividad Física", ["Bajo", "Moderado", "Alto"])
    alcohol = st.sidebar.selectbox("Consumo de Alcohol", ["Nunca", "Ocasional", "Moderado", "Frecuente"])
    sex = st.sidebar.selectbox("Sexo", ["Male", "Female"])
    ethnicity = st.sidebar.selectbox("Etnia", ["White", "Black", "Asian", "Hispanic", "Other"])
    family = st.sidebar.selectbox("Antecedentes Familiares de Diabetes", ["Yes", "No"])

    data = {
        'Age': age,
        'BMI': bmi,
        'Fasting_Blood_Glucose': glucose,
        'Physical_Activity_Level': physical,
        'Alcohol_Consumption': alcohol,
        'Sex': sex,
        'Ethnicity': ethnicity,
        'Family_History_of_Diabetes': 1 if family == "Yes" else 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("🔎 Datos del Paciente")
st.write(input_df)

# Predicción
if st.button("📊 Predecir Riesgo de Diabetes"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Resultado de la Predicción")
    st.success("**RIESGO ALTO DE DIABETES**" if prediction == 1 else "**RIESGO BAJO DE DIABETES**")
    st.metric("Probabilidad estimada", f"{proba:.2%}")

# Visualización de insights (simulados para visualización)
if st.checkbox("📊 Mostrar análisis visual (simulado)"):
    # Datos ficticios
    df = pd.DataFrame({
        'Age': np.random.normal(45, 10, 300),
        'BMI': np.random.normal(28, 5, 300),
        'HbA1c': np.random.normal(5.8, 1.0, 300),
        'Fasting_Blood_Glucose': np.random.normal(100, 25, 300),
        'Sex': np.random.choice(["Male", "Female"], 300),
        'Alcohol_Consumption': np.random.choice(["Nunca", "Ocasional", "Moderado", "Frecuente"], 300)
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribución de HbA1c")
        fig, ax = plt.subplots()
        sns.histplot(df['HbA1c'], kde=True, bins=30, ax=ax, color='skyblue')
        st.pyplot(fig)

    with col2:
        st.markdown("### Relación BMI vs Glucosa en Ayunas")
        fig2, ax2 = plt.subplots()
        sns.regplot(x="BMI", y="Fasting_Blood_Glucose", data=df, ax=ax2)
        st.pyplot(fig2)
