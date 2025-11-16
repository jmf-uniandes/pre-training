import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =============================
# CONFIGURACIÓN DE LA PÁGINA
# =============================
st.set_page_config(
    page_title="Buscador de Hits 🎵",
    page_icon="🎵",
    layout="wide"
)

# ========== CARGAR CSS ==========
css_path = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =============================
# RUTAS Y ARCHIVOS
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

API_URL = "http://127.0.0.1:8000/songs/predict_hit"
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "spotify_clean_modeling.csv")

df = pd.read_csv(DATA_PATH)

# =============================
# TÍTULO PREMIUM
# =============================
st.markdown(
    """
    <h1 style='text-align:center; color:#32F5C8;'>🎵 EL BUSCADOR DE HITS</h1>
    <h3 style='text-align:center; color: #7FFFD4; margin-top:-15px;'>
        Crea tu Receta para el Éxito Musical
    </h3>
    """,
    unsafe_allow_html=True
)

st.write("")

# =============================
# LAYOUT PRINCIPAL
# =============================
col1, col2 = st.columns([1.2, 1.8])

# =====================================
# COLUMNA IZQUIERDA — SLIDERS
# =====================================
with col1:

    st.subheader("Ajusta los atributos de la canción")

    genre = st.selectbox("Género", sorted(df["genre"].unique()))

    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0, 0.1)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.001)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.01)
    valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0, 1.0)
    duration_ms = st.slider("Duración (ms)", 30000, 400000, 180000, 1000)

    if st.button("🎯 Predecir HIT", use_container_width=True):

        payload = {
            "genre": genre,
            "acousticness": acousticness,
            "danceability": danceability,
            "energy": energy,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "loudness": loudness,
            "speechiness": speechiness,
            "tempo": tempo,
            "valence": valence,
            "duration_ms": duration_ms
        }

        response = requests.post(API_URL, json=payload)
        data = response.json()

        prob = data["hit_probability"]
        pred = data["hit_prediction"]

        st.session_state["pred_prob"] = prob
        st.session_state["pred_label"] = pred


# =====================================
# COLUMNA DERECHA — RESULTADO PREMIUM
# =====================================
with col2:

    st.subheader("Resultado de la Predicción")

    if "pred_prob" in st.session_state:

        prob = st.session_state["pred_prob"]
        pred = st.session_state["pred_label"]

        # ===== Velocímetro GAUGE =====
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#32F5C8"},
                'steps': [
                    {'range': [0, 40], 'color': "#1A1A1A"},
                    {'range': [40, 70], 'color': "#333333"},
                    {'range': [70, 100], 'color': "#0F3F34"}
                ],
            }
        ))

        gauge.update_layout(height=320, margin=dict(t=30, b=10))

        st.plotly_chart(gauge, use_container_width=True)

        # Texto de interpretación
        if prob > 0.70:
            st.success("🔥 **Potencial de Éxito: ALTO**")
        elif prob > 0.40:
            st.warning("🎧 **Potencial de Éxito: MEDIO**")
        else:
            st.error("❄️ **Potencial de Éxito: BAJO**")

        # Distribución comparativa de probabilidades reales
        st.write("### Distribución de Probabilidades Reales")

        hist = go.Figure()
        hist.add_trace(go.Histogram(
            x=df["is_hit"],
            marker_color="#32F5C8"
        ))
        hist.update_layout(height=250)
        st.plotly_chart(hist, use_container_width=True)

    else:
        st.info("Configura los sliders y presiona **Predecir HIT**.")


