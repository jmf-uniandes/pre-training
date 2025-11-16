from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes.songs import router as songs_router

app = FastAPI(
    title="Spotify Hit Predictor API",
    version="1.0.0"
)


app.mount("/static", StaticFiles(directory="src/api/static"), name="static")


@app.get("/favicon.ico")
def favicon():
    return FileResponse("src/api/static/favicon.ico")


# Rutas de la API
app.include_router(songs_router, prefix="/songs", tags=["Songs"])

@app.get("/")
def root():
    return {"message": "API running"}
