"""
Módulo: routes.py

Responsabilidad:
----------------
Definir la aplicación FastAPI que se expondrá como servicio,
incluyendo:

- Endpoint de salud (/health).
- Endpoint principal de predicción (/songs/predict_hit).

Este archivo será el punto de entrada para uvicorn:

    uvicorn src.api.routes:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model_service import SongFeatures, predict_hit_probability

# ---------------------------------------------------------
# Creación de la aplicación FastAPI
# ---------------------------------------------------------
app = FastAPI(
    title="Buscador de Hits - API",
    description=(
        "API para clasificar canciones en 'hit' o 'no hit' "
        "basada en atributos de audio de Spotify."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------
# Configuración de CORS
# ---------------------------------------------------------
# Permitimos peticiones desde cualquier origen durante desarrollo.
# En producción, se recomienda restringir a dominios específicos.
app.add_middleware(
    CORSMiddleware,
    # Cambiar por lista de orígenes permitidos en prod
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health", tags=["system"])
def health_check():
    """
    Endpoint de salud para verificar que el servicio está vivo.

    Returns
    -------
    dict
        Un diccionario simple con el estado.
    """
    return {"status": "ok"}


@app.post("/songs/predict_hit", tags=["predictions"])
def predict_hit(song: SongFeatures):
    """
    Endpoint principal para predecir si una canción será un hit.

    Request body:
    -------------
    Un JSON con los atributos definidos en SongFeatures.

    Response:
    ---------
    JSON con:
    - probability_hit: probabilidad de hit (0–1).
    - is_hit: clasificación booleana usando un umbral (por defecto 0.5).
    - threshold: valor del umbral utilizado.
    """
    proba = predict_hit_probability(song)
    threshold = 0.5
    is_hit = proba >= threshold

    return {
        "probability_hit": proba,
        "is_hit": is_hit,
        "threshold": threshold,
    }
