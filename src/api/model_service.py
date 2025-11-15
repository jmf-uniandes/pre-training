"""
Módulo: model_service.py

Responsabilidad:
----------------
1. Definir el esquema de entrada de datos de una canción
   utilizando Pydantic (SongFeatures).
2. Cargar en memoria el modelo entrenado (pipeline de scikit-learn).
3. Exponer funciones auxiliares para convertir datos y predecir
   la probabilidad de que una canción sea un 'hit'.
"""

from __future__ import annotations

import functools
import os
from typing import Dict, Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field, validator


class SongFeatures(BaseModel):
    """
    Esquema de entrada para el endpoint /songs/predict_hit.

    Cada atributo está documentado para que FastAPI genere
    una documentación clara en /docs (Swagger UI).
    """

    genre: str = Field(
        ...,
        description="Género musical de la canción (ej. pop, rock, latin).",
        example="pop",
    )
    acousticness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proporción de componente acústica: 0 = nada, 1 = totalmente acústica.",
        example=0.12,
    )
    danceability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Qué tan bailable es la canción: 0 = nada bailable, 1 = muy bailable.",
        example=0.8,
    )
    duration_ms: int = Field(
        ...,
        ge=10000,
        description="Duración de la canción en milisegundos.",
        example=180000,
    )
    energy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Intensidad y actividad percibida: 0 = muy baja, 1 = muy alta.",
        example=0.75,
    )
    instrumentalness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad de que no tenga voces.",
        example=0.0,
    )
    liveness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad de que haya público en vivo.",
        example=0.1,
    )
    loudness: float = Field(
        ...,
        description="Intensidad en decibelios (valores negativos, p.ej. -6.0 dB).",
        example=-5.0,
    )
    mode: int = Field(
        ...,
        ge=0,
        le=1,
        description="Modo musical: 1 = mayor, 0 = menor.",
        example=1,
    )
    speechiness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proporción de contenido hablado.",
        example=0.05,
    )
    tempo: float = Field(
        ...,
        description="Tempo estimado de la pista en BPM (beats per minute).",
        example=120.0,
    )
    time_signature: int = Field(
        ...,
        ge=1,
        le=12,
        description="Compás estimado de la pista (ej. 4 = 4/4).",
        example=4,
    )
    valence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Qué tan positiva/feliz suena la canción: 0 = triste, 1 = muy feliz.",
        example=0.9,
    )

    @validator("genre")
    def genre_no_vacio(cls, v: str) -> str:
        """
        Valida que el género no sea una cadena vacía.
        """
        v = v.strip()
        if not v:
            raise ValueError("El campo 'genre' no puede estar vacío.")
        return v


@functools.lru_cache(maxsize=1)
def load_pipeline() -> Any:
    """
    Carga el pipeline de clasificación de hits desde la carpeta `models`.

    - Utiliza lru_cache para que el modelo se cargue SOLO una vez
      durante la vida del proceso del servidor.
    - Se asume que el archivo fue generado por los notebooks
      (especialmente 04_model_training.ipynb).

    Returns
    -------
    pipeline : objeto scikit-learn compatible con .predict_proba()
    """
    # Localizar raíz del proyecto y construir ruta al archivo del modelo
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(root, "models", "spotify_hit_classifier.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en: {model_path}\n"
            "Asegúrate de haber ejecutado el notebook de entrenamiento "
            "y guardado el modelo en esa ruta."
        )

    pipeline = joblib.load(model_path)
    return pipeline


def song_to_dataframe(song: SongFeatures) -> pd.DataFrame:
    """
    Convierte una instancia SongFeatures en un DataFrame con UNA sola fila.

    Esto es necesario porque el pipeline fue entrenado con un DataFrame
    de múltiples filas, y scikit-learn espera el mismo tipo de estructura
    en la fase de predicción.

    Parameters
    ----------
    song : SongFeatures
        Objeto con los atributos de audio de una canción.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame con una fila y las columnas en el orden esperado por el modelo.
    """
    data: Dict[str, Any] = song.dict()

    # IMPORTANTE: el orden de las columnas debe coincidir con
    # el orden que se usó en el entrenamiento del pipeline.
    columns_order = [
        "genre",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "mode",
        "speechiness",
        "tempo",
        "time_signature",
        "valence",
    ]

    # Creamos el DataFrame forzando el orden de columnas
    df = pd.DataFrame([data], columns=columns_order)
    return df


def predict_hit_probability(song: SongFeatures) -> float:
    """
    Calcula la probabilidad de que una canción sea un hit.

    Pasos:
    ------
    1. Cargar el pipeline (si no está ya en memoria).
    2. Convertir la canción (SongFeatures) en un DataFrame de una fila.
    3. Usar pipeline.predict_proba(df) para obtener la probabilidad
       de la clase positiva (1 = hit).

    Parameters
    ----------
    song : SongFeatures
        Objeto con los atributos de audio.

    Returns
    -------
    proba : float
        Probabilidad (entre 0 y 1) de que la canción sea un hit.
    """
    pipeline = load_pipeline()
    df = song_to_dataframe(song)
    proba = pipeline.predict_proba(df)[0, 1]
    return float(proba)
