import os
import pandas as pd
import streamlit as st


# Ruta utils.py  → PRE-TRAINING/src/dashboard/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subir 1 nivel → PRE-TRAINING/src
SRC_DIR = os.path.dirname(BASE_DIR)

# Subir 1 nivel → PRE-TRAINING
ROOT_DIR = os.path.dirname(SRC_DIR)


# RUTAS DE ARCHIVOS
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "spotify_clean_modeling.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "custom.css")
# Local
# API_URL = "http://127.0.0.1:8000/songs/predict_hit"
API_URL = "https://pre-training-production.up.railway.app/songs/predict_hit"



# FUNCIONES
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

