"""
Módulo: buscador_de_hits.py

Responsabilidad:
----------------
Definir la aplicación de Streamlit que actuará como dashboard:

- Pestaña de EDA interactivo del dataset de Spotify.
- Pestaña "Buscador de Hits" donde el usuario construye una canción
  con sliders y el sistema consulta la API para predecir si será un hit.

Ejecución local:
----------------
Desde la raíz del proyecto, ejecutar:

    streamlit run src/dashboard/buscador_de_hits.py

En Docker, este archivo será el entrypoint del contenedor del dashboard.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.utils.data_loader import load_clean_spotify
from src.utils.eda_helpers import top_genres_by_hits, numeric_corr_heatmap


# ---------------------------------------------------------
# Configuración de la página de Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Buscador de Hits",
    page_icon="🎧",
    layout="wide",
)


# URL base de la API (se puede sobreescribir con variable de entorno)
API_BASE_URL = os.getenv("HITS_API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/songs/predict_hit"


# ---------------------------------------------------------
# Funciones auxiliares de caché
# ---------------------------------------------------------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """
    Carga el dataset limpio de Spotify usando el módulo de utilidades.

    Se cachea con @st.cache_data para evitar recargar el CSV en cada
    interacción del usuario.
    """
    df = load_clean_spotify()
    return df


def layout_header():
    """
    Dibuja el encabezado del dashboard con título y descripción breve.
    """
    st.title("🎧 Buscador de Hits")
    st.caption(
        "Análisis de atributos musicales y predicción de popularidad de canciones")
    st.markdown(
        """
        Este dashboard tiene dos secciones principales:

        1. **Exploración de datos (EDA)**  
           Observa cómo se distribuyen los atributos de audio y qué
           diferencias hay entre canciones 'hit' y 'no hit'.

        2. **Buscador de Hits**  
           Ajusta los sliders para definir la "receta" de una canción
           y consulta en tiempo real la API para ver la probabilidad
           de que sea un éxito.
        """
    )


# ---------------------------------------------------------
# Pestaña EDA
# ---------------------------------------------------------
def tab_eda(df: pd.DataFrame):
    """
    Construye la pestaña de Exploración de Datos.

    Componentes:
    -----------
    - Filtros por género y rango de popularidad.
    - KPIs principales.
    - Histogramas de popularidad.
    - Ranking de géneros por número de hits.
    - Mapa de calor de correlaciones entre atributos numéricos.
    """
    st.subheader("Exploración interactiva de canciones")

    col1, col2 = st.columns(2)

    with col1:
        generos = sorted(df["genre"].dropna().unique().tolist())
        genero_sel = st.multiselect(
            "Géneros musicales",
            options=generos,
            default=generos[:5] if len(generos) >= 5 else generos,
        )

    with col2:
        min_pop = int(df["popularity"].min())
        max_pop = int(df["popularity"].max())
        rango_pop = st.slider(
            "Rango de popularidad",
            min_value=min_pop,
            max_value=max_pop,
            value=(min_pop, max_pop),
        )

    # Filtro compuesto por género(s) y rango de popularidad
    mask = df["popularity"].between(rango_pop[0], rango_pop[1])
    if genero_sel:
        mask &= df["genre"].isin(genero_sel)

    df_filtrado = df[mask]

    if df_filtrado.empty:
        st.warning("No hay canciones que cumplan con los filtros seleccionados.")
        return

    # KPIs básicos
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.metric("Total canciones", f"{len(df_filtrado):,}")

    with col_kpi2:
        st.metric(
            "Porcentaje de hits",
            f"{df_filtrado['is_hit'].mean() * 100:.2f} %",
        )

    with col_kpi3:
        st.metric(
            "Popularidad media",
            f"{df_filtrado['popularity'].mean():.1f}",
        )

    with col_kpi4:
        st.metric(
            "Géneros únicos",
            f"{df_filtrado['genre'].nunique()}",
        )

    st.markdown("---")

    # Gráficos
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.write("#### Distribución de popularidad (según filtros)")
        fig_pop = px.histogram(
            df_filtrado,
            x="popularity",
            nbins=40,
            color="is_hit",
            labels={"is_hit": "Es hit"},
            marginal="box",
            title="Distribución de popularidad por condición de hit",
        )
        st.plotly_chart(fig_pop, use_container_width=True)

    with col_g2:
        st.write("#### Hits por género")
        fig_hits_genre = top_genres_by_hits(df_filtrado, top_n=15)
        st.plotly_chart(fig_hits_genre, use_container_width=True)

    st.markdown("---")

    st.write("#### Mapa de calor de correlaciones (atributos numéricos)")
    numeric_cols: List[str] = [
        "popularity",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_filtrado.columns]

    if len(numeric_cols) > 1:
        fig_corr = numeric_corr_heatmap(df_filtrado, numeric_cols)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No hay suficientes columnas numéricas para mostrar correlaciones.")


# ---------------------------------------------------------
# Pestaña Buscador de Hits
# ---------------------------------------------------------
def tab_predictor():
    """
    Construye la pestaña principal "Buscador de Hits".

    Componentes:
    -----------
    - Sliders y controles para definir los atributos de audio de una pista.
    - Botón para enviar la solicitud a la API.
    - Visualización de la probabilidad de hit (métrica y gráfico).
    """
    st.subheader("Diseña tu canción y predice si será un hit")

    col_form, col_res = st.columns([1.3, 1])

    with col_form:
        st.markdown("##### Atributos de la canción")

        genre = st.text_input("Género", value="pop")

        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3, 0.01)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.7, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.8, 0.01)
        valence = st.slider("Valence (positividad)", 0.0, 1.0, 0.6, 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0, 0.5)
        tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0, 1.0)
        duration_ms = st.slider("Duración (ms)", 30_000,
                                420_000, 180_000, 1_000)

        mode = st.radio(
            "Modo",
            [1, 0],
            format_func=lambda m: "Mayor (1)" if m == 1 else "Menor (0)",
            horizontal=True,
        )

        time_signature = st.slider(
            "Compás (time_signature)",
            min_value=1,
            max_value=12,
            value=4,
            step=1,
        )

        if st.button("Predecir probabilidad de hit"):
            payload: Dict[str, Any] = {
                "genre": genre,
                "acousticness": acousticness,
                "danceability": danceability,
                "duration_ms": duration_ms,
                "energy": energy,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "loudness": loudness,
                "mode": mode,
                "speechiness": speechiness,
                "tempo": tempo,
                "time_signature": time_signature,
                "valence": valence,
            }

            try:
                resp = requests.post(
                    PREDICT_ENDPOINT, json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Error al llamar a la API: {e}")
                return

            proba = data.get("probability_hit", 0.0)
            is_hit = data.get("is_hit", False)

            with col_res:
                st.write("### Resultado")
                st.metric(
                    label="Probabilidad de HIT",
                    value=f"{proba * 100:.1f} %",
                )

                if is_hit:
                    st.success(
                        "Según el modelo, esta canción tiene **alta probabilidad de ser un hit**."
                    )
                else:
                    st.warning(
                        "Según el modelo, esta canción **probablemente no será un hit**."
                    )

                # Pequeño gráfico circular para visualizar la probabilidad
                fig_polar = px.bar_polar(
                    r=[proba * 100, (1 - proba) * 100],
                    theta=["Hit", "No hit"],
                    range_r=[0, 100],
                    title="Probabilidad de hit vs no hit",
                )
                st.plotly_chart(fig_polar, use_container_width=True)
        else:
            with col_res:
                st.info(
                    "Configura los sliders y pulsa el botón para obtener una predicción.")


# ---------------------------------------------------------
# Función principal de la app Streamlit
# ---------------------------------------------------------
def main():
    df = cargar_datos()
    layout_header()

    tab1, tab2 = st.tabs(["Exploración de datos", "Buscador de hits"])

    with tab1:
        tab_eda(df)

    with tab2:
        tab_predictor()


if __name__ == "__main__":
    main()
