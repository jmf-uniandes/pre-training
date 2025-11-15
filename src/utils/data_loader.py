"""
Módulo: data_loader.py

Responsabilidad:
----------------
Centralizar toda la lógica de rutas y carga/guardado de datos
para no “romper” las rutas cuando cambiamos de entorno
(Jupyter, FastAPI, Streamlit, Docker, etc.).

Todas las funciones aquí asumen que se ejecutan desde cualquier
parte del proyecto, pero siempre localizan la raíz del repo
(la carpeta CASE-STUDY-SPOTIFY) de forma relativa a este archivo.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def get_project_root() -> str:
    """
    Devuelve la ruta absoluta a la raíz del proyecto.

    Implementación:
    ---------------
    - __file__ → ruta de ESTE archivo (src/utils/data_loader.py).
    - subimos dos niveles: utils/ → src/ → raíz del repo.

    Returns
    -------
    root : str
        Ruta absoluta de la carpeta raíz del proyecto.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_data_paths() -> Tuple[str, str]:
    """
    Construye las rutas a los archivos de datos crudos y procesados.

    Returns
    -------
    raw_path : str
        Ruta a data/raw/SpotifyFeatures.csv
    clean_path : str
        Ruta a data/processed/spotify_clean.csv
    """
    root = get_project_root()
    raw_path = os.path.join(root, "data", "raw", "SpotifyFeatures.csv")
    clean_path = os.path.join(root, "data", "processed", "spotify_clean.csv")
    return raw_path, clean_path


def load_raw_spotify() -> pd.DataFrame:
    """
    Carga el dataset crudo original de Spotify (tal como se descargó de Kaggle).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame con los datos crudos.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta esperada.
    """
    raw_path, _ = get_data_paths()
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo crudo en: {raw_path}")

    df = pd.read_csv(raw_path)
    print(
        f"✅ Datos crudos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def load_clean_spotify() -> pd.DataFrame:
    """
    Carga el dataset procesado (output de 01_loader.ipynb).

    Este archivo debe contener, al menos:
    - columna de popularidad: 'popularity'
    - variable objetivo binaria: 'is_hit'
    - atributos de audio: danceability, energy, valence, etc.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame con los datos limpios.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta esperada.
    """
    _, clean_path = get_data_paths()
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"No se encontró el archivo limpio en: {clean_path}")

    df = pd.read_csv(clean_path)
    print(
        f"✅ Datos limpios cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def save_clean_spotify(df: pd.DataFrame) -> str:
    """
    Guarda un DataFrame como archivo CSV en data/processed/spotify_clean.csv

    Parámetros típicos:
    -------------------
    Esta función se utiliza al final de 01_loader.ipynb,
    una vez que se han aplicado todas las transformaciones de limpieza
    y se ha creado la variable objetivo 'is_hit'.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame ya limpio y listo para modelado.

    Returns
    -------
    path : str
        Ruta final donde se guardó el archivo.
    """
    _, clean_path = get_data_paths()
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f"💾 Dataset limpio guardado en: {clean_path}")
    return clean_path
