import streamlit as st
import requests
import numpy as np
import pandas as pd

# Utilidades y gauge modular
from utils import load_dataset, load_css, API_URL
from gauge import create_gauge_chart


# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================
st.set_page_config(
    page_title="Buscador de Hits 🎵",
    page_icon="🎵",
    layout="wide"
)

# Cargar estilos globales
load_css()

# Dataset para cargar géneros
df = load_dataset()


# ============================================================
# TÍTULO PRINCIPAL
# ============================================================
st.markdown(
    """
    <h1 style='text-align:center; color:#32F5C8;'>🎵 EL BUSCADOR DE HITS</h1>
    <h3 style='text-align:center; color:#7FFFD4; margin-top:-15px;'>
        Crea tu Receta para el Éxito Musical
    </h3>
    """,
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
col1, col2 = st.columns([1.2, 1.8])


# ============================================================
# COLUMNA IZQUIERDA — SLIDERS
# ============================================================
with col1:

    st.subheader("🎚 Ajusta los atributos de la canción")

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

        st.session_state["pred_prob"] = data["hit_probability"]
        st.session_state["pred_label"] = data["hit_prediction"]



# ============================================================
# COLUMNA DERECHA — RESULTADO PREMIUM
# ============================================================
with col2:

    st.subheader("📈 Resultado de la Predicción")

    if "pred_prob" in st.session_state:

        prob = float(st.session_state["pred_prob"])
        pred = int(st.session_state["pred_label"])
        prob_pct = int(prob * 100)

        # ====================================================
        # GAUGE PREMIUM PLOTLY
        # ====================================================
        gauge_fig = create_gauge_chart(prob_pct, "PROBABILIDAD DE HIT")
        # CENTRAR EL GAUGE CON COLUMNAS
        g1, g2, g3 = st.columns([1, 2, 1])  # columna central 2x más grande
        with g2:
            st.plotly_chart(gauge_fig, use_container_width=True)
 
        # ====================================================
        # INTERPRETACIÓN
        # ====================================================
        st.write("")

        if pred == 1:
            if prob >= 0.85:
                st.success("🔥 **HIT Seguro — Altísima confianza del modelo**")
            elif prob >= 0.70:
                st.success("🎵 **HIT Probable — Buena confianza del modelo**")
            else:
                st.warning("🎧 **HIT Débil — Baja confianza del modelo**")
        else:
            if prob <= 0.15:
                st.error("❄️ **NO HIT — Muy seguro**")
            elif prob <= 0.30:
                st.warning("⚠️ **NO HIT Probable — Señal débil**")
            else:
                st.info("ℹ️ **NO HIT — Indeciso**")


        # ====================================================
        # ESPECTRO DE PROBABILIDAD (GRÁFICO)
        # ====================================================
        st.markdown("### 📊 Espectro de Probabilidad")
        st.caption("Distribución centrada en tu probabilidad")

        x = np.linspace(0, 1, 400)
        y = np.exp(-((x - prob) ** 2) / 0.003)

        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="#32F5C8", width=3),
            fill="tozeroy",
            fillcolor="rgba(50,245,200,0.15)"
        ))

        fig.add_vline(x=prob, line_color="red", line_width=4)

        fig.update_layout(
            height=260,
            margin=dict(t=10, b=10),
            xaxis_title="Probabilidad",
            yaxis_title="Intensidad relativa",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Configura los sliders y presiona **Predecir HIT**.")
