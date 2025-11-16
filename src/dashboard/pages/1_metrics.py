import streamlit as st
import os
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# ================================
# RUTAS
# ================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "spotify_clean_modeling.csv")



# ================================
# DATASET
# ================================
df = pd.read_csv(DATA_PATH)

# ================================
# UI
# ================================
st.markdown(
    """
    <h1 style='color:#32F5C8;'>📊 Métricas del Dataset Spotify</h1>
    <h4 style='color:#7FFFD4; margin-top:-10px;'>
        Análisis general del dataset utilizado para entrenar tu modelo de predicción de hits.
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ================================
# Vista general
# ================================
st.subheader("📌 Vista general")
st.dataframe(df.head())

# ================================
# Estadísticas principales
# ================================
st.subheader("📌 Estadísticas numéricas")
st.dataframe(df.describe().T)

# ================================
# Distribución de Géneros
# ================================
st.subheader("🎼 Canciones por género")

genre_counts = df["genre"].value_counts()
fig = px.bar(
    genre_counts,
    x=genre_counts.index,
    y=genre_counts.values,
    labels={"x": "Género", "y": "Cantidad"},
    title="Cantidad de Canciones por Género",
    color=genre_counts.values,
    color_continuous_scale="agsunset"
)
fig.update_layout(showlegend=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# ================================
# Distribución de HIT vs NO-HIT
# ================================
st.subheader("🔥 Distribución HIT vs NO-HIT")

hit_counts = df["is_hit"].value_counts()
fig2 = px.pie(
    values=hit_counts.values,
    names=["NO HIT", "HIT"],
    title="Proporción de Canciones HIT",
    color=hit_counts.values,
    color_discrete_sequence=["#1a1a1a", "#32F5C8"]
)
fig2.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig2, use_container_width=True)

st.write("---")
st.success("Página cargada correctamente.")
