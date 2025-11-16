from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import os

from src.api.utils.preprocess import preprocess_input_features

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_pipeline.joblib")

model = joblib.load(MODEL_PATH)


class SongFeatures(BaseModel):
    genre: str
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    tempo: float
    valence: float
    duration_ms: float


@router.post("/predict_hit")
def predict_hit(song: SongFeatures):

    df = preprocess_input_features(song)
    prob = model.predict_proba(df)[0][1]

    return {
        "hit_probability": float(prob),
        "hit_prediction": 1 if prob >= 0.64 else 0
    }
