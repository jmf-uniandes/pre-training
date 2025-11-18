import os
import pandas as pd
import streamlit as st


# Ruta utils.py  → PRE-TRAINING/src/dashboard/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta → PRE-TRAINING/src
SRC_DIR = os.path.dirname(BASE_DIR)

# Ruta → PRE-TRAINING
ROOT_DIR = os.path.dirname(SRC_DIR)


# Rutas de Archivos
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "spotify_clean_modeling.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "custom.css")
# Local
# API_URL = "http://127.0.0.1:8000/songs/predict_hit"
API_URL = "https://pre-training-production.up.railway.app/songs/predict_hit"

# Diccionario global de descripciones para sliders y atributos
DESCRIPTIONS = {
    "genre": "Género musical",
    "acousticness": "Qué tan acústica es la canción",
    "danceability": "Qué tan bailable es la canción",
    "energy": "Intensidad y actividad percibida",
    "loudness": "Volumen promedio (dB)",
    "speechiness": "Presencia de palabras habladas",
    "instrumentalness": "Nivel instrumental (sin voz)",
    "liveness": "Probabilidad de grabación en vivo",
    "valence": "Qué tan positiva/feliz suena",
    "tempo": "Velocidad del ritmo (BPM)",
    "duration_ms": "Duración de la canción en milisegundos"
}


# FUFunciones utilitarias
@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)

def load_css():
    """Inyecta CSS global en el DOM correcto de Streamlit (multipage compatible)."""
    with open(CSS_PATH, "r") as f:
        css = f.read()

    st.markdown(
        f"""
        <style>
        {css}
        </style>
        """,
        unsafe_allow_html=True
    )

