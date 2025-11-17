import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import shap
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# CONFIGURACIÓN GENERAL
# =============================
st.set_page_config(
    page_title="Buscador de Hits – Spotify ML",
    page_icon="🎵",
    layout="wide"
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
ROOT_DIR = os.path.dirname(BASE_DIR)                                    # .../pre-training
API_URL = "http://127.0.0.1:8000/songs/predict_hit"
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "spotify_clean_modeling.csv")
MODEL_PATH = os.path.join(BASE_DIR, "api", "models", "model_pipeline.joblib")


# print(BASE_DIR)
# print(DATA_PATH)
# print (MODEL_PATH)
# print(ROOT_DIR)
# print(os.path.exists(DATA_PATH))


st.title("🎵 Buscador de Hits – Dashboard Avanzado")
st.write("Ajusta los atributos sonoros basados en la estructura REAL del dataset SpotifyFeatures.")

# =============================
# CARGAR MODELO PARA SHAP
# =============================

model = joblib.load(MODEL_PATH)

# =============================
# CARGAR DATASET PARA HISTOGRAMAS
# =============================
df = pd.read_csv(DATA_PATH)

# ==========================================================
# 1) INPUTS EXACTOS DEL DATASET
# ==========================================================

GENRES = [
    "Pop", "Rap", "Dance", "Hip-Hop",
    "Reggaeton", "R&B", "Electronic",
    "Indie", "Rock"
]

col1, col2 = st.columns(2)

with col1:
    genre = st.selectbox("Género musical (categoría)", GENRES)

    danceability = st.slider("Danceability (0–1)", 0.0, 1.0, 0.50)
    energy = st.slider("Energy (0–1)", 0.0, 1.0, 0.50)
    valence = st.slider("Valence (0–1)", 0.0, 1.0, 0.50)
    acousticness = st.slider("Acousticness (0–1)", 0.0, 1.0, 0.30)
    instrumentalness = st.slider("Instrumentalness (0–1)", 0.0, 1.0, 0.00)

with col2:
    liveness = st.slider("Liveness (0–1)", 0.0, 1.0, 0.20)
    speechiness = st.slider("Speechiness (0–1)", 0.0, 1.0, 0.10)
    tempo = st.slider("Tempo (40–220 BPM)", 40.0, 220.0, 120.0)
    loudness = st.slider("Loudness (dB) (-60 → 0)", -60.0, 0.0, -10.0)
    duration_ms = st.slider("Duración (ms)", 30000, 500000, 180000)

# Construir payload EXACTO para la API
payload = {
    "genre": genre,
    "danceability": danceability,
    "energy": energy,
    "valence": valence,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "speechiness": speechiness,
    "tempo": tempo,
    "loudness": loudness,
    "duration_ms": duration_ms
}

# ==========================================================
# 2) LLAMAR API
# ==========================================================
prob = None
is_hit = None

if st.button("Predecir HIT 🚀"):
    response = requests.post(API_URL, json=payload)
    data = response.json()

    prob = data["hit_probability"]
    is_hit = data["hit_prediction"]


    st.metric("Probabilidad de HIT", f"{prob*100:.2f}%")

    if is_hit == 1:
        st.success("Esta canción tiene perfil de HIT 🎉")
    else:
        st.warning("No parece un HIT")

# ==========================================================
# 3) ESPECTRO DE PROBABILIDAD
# ==========================================================
if prob is not None:
    st.subheader("📊 Espectro de probabilidad")

    x_vals = np.linspace(0, 1, 200)
    y_vals = np.exp(-(x_vals - prob)**2 / 0.002)

    fig_prob = px.line(
        x=x_vals,
        y=y_vals,
        title="Distribución centrada en tu probabilidad",
        labels={"x": "Probabilidad", "y": "Intensidad relativa"}
    )

    fig_prob.add_vline(x=prob, line_color="red", line_width=3)
    fig_prob.update_layout(height=300)

    st.plotly_chart(fig_prob, use_container_width=True)

# ==========================================================
# 4) HISTOGRAMAS COMPARATIVOS
# ==========================================================
st.header("🎼 Comparación con HITS reales")

if "is_hit" in df.columns:
    hits = df[df["is_hit"] == 1]

    feature = st.selectbox(
        "Selecciona feature para comparar",
        ["danceability", "energy", "valence", "acousticness",
         "instrumentalness", "liveness", "speechiness", "tempo"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(hits[feature], kde=True, color="blue", label="HITS", ax=ax)
    ax.axvline(payload[feature], color="red", lw=3, label="Tu canción")
    ax.set_title(f"Distribución de {feature} en canciones HIT")
    ax.legend()
    st.pyplot(fig)

# ==========================================================
# 5) EXPLICABILIDAD SHAP
# ==========================================================
st.header("🧠 Explicabilidad del modelo (SHAP)")

try:
    explainer = shap.TreeExplainer(model["model"])

    df_sample = pd.DataFrame([payload])
    X_proc = model["preprocess"].transform(df_sample)

    shap_values = explainer.shap_values(X_proc)

    st.subheader("Importancia de atributos para esta predicción")

    shap_fig = shap.force_plot(
        explainer.expected_value,
        shap_values,
        X_proc,
        matplotlib=True,
        show=False
    )

    st.pyplot(shap_fig)

except Exception as e:
    st.error(f"Error generando SHAP: {e}")
    st.info("SHAP funciona con modelos basados en árboles (LightGBM, XGBoost, etc).")
