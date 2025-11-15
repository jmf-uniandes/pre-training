"""
Módulo: eda_helpers.py

Responsabilidad:
----------------
Contener funciones de apoyo para visualizaciones EDA,
que se utilizan tanto en el notebook 02_eda.ipynb como
en el dashboard de Streamlit.

Aquí se definen funciones que producen gráficos de Plotly
o agregaciones estadísticas listas para mostrarse.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px


def top_genres_by_hits(df: pd.DataFrame, top_n: int = 15) -> px.bar:
    """
    Construye una figura de Plotly que muestra los géneros
    con mayor número de canciones 'hit'.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con al menos columnas 'genre' e 'is_hit'.
    top_n : int, optional
        Número de géneros a mostrar, por defecto 15.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figura lista para mostrarse con st.plotly_chart.
    """
    df_hits = df[df["is_hit"] == 1]
    resumen = (
        df_hits.groupby("genre")["is_hit"]
        .count()
        .reset_index()
        .rename(columns={"is_hit": "num_hits"})
        .sort_values("num_hits", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        resumen,
        x="genre",
        y="num_hits",
        title=f"Top {top_n} géneros por número de hits",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def numeric_corr_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> px.imshow:
    """
    Genera un mapa de calor de correlaciones para columnas numéricas.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas numéricas.
    numeric_cols : list of str
        Lista de nombres de columnas numéricas a correlacionar.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Heatmap de correlaciones.
    """
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlaciones entre atributos de audio",
    )
    return fig
