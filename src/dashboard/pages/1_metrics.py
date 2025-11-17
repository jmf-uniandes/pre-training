import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_dataset, load_css


# Configuración de la página
st.set_page_config(
    page_title="Métricas del Dataset",
    page_icon="📊",
    layout="wide"
)
# carga de CSS personalizado
load_css()

# Carga de datos
df = load_dataset()
generos_ordenados = sorted(df["genre"].unique())

# Título y descripción
st.markdown(
    """
    <h1 style='color:#32F5C8;'>📊 Métricas del Dataset Spotify</h1>
    <h4 style='color:#7FFFD4; margin-top:-10px;'>
        Exploración avanzada del dataset utilizado por el modelo de predicción
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("---")


# 1 Vista previa del dataset y estadísticas descriptivas

with st.container(border=True):
    st.subheader("📌 Vista previa del dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📌 Estadísticas descriptivas")
    st.dataframe(df.describe().T, use_container_width=True)


# 2 Gráficos generales por género

with st.container(border=True):
    st.subheader("🎼 Distribución por Género")

    genre_counts = df["genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]

    fig1 = px.bar(
        genre_counts,
        x="genre",
        y="count",
        title="Distribución de Canciones por Género",
        color="count",
        color_continuous_scale="agsunset",
    )
    st.plotly_chart(fig1, use_container_width=True)

    
    # HIT por género    
    st.subheader("🔥 Canciones HIT por Género")

    hits_por_genero = (
        df.groupby("genre")["is_hit"]
        .sum()
        .reset_index()
        .sort_values(by="is_hit", ascending=False)
    )

    fig_hits_genre = px.bar(
        hits_por_genero,
        x="genre",
        y="is_hit",
        title="Cantidad de Canciones HIT por Género",
        color="is_hit",
        color_continuous_scale="teal"
    )
    st.plotly_chart(fig_hits_genre, use_container_width=True)

    
    # HIT RATE por género
    
    st.subheader("📈 HIT RATE por Género (hits / total)")

    hit_rate = hits_por_genero.merge(genre_counts, on="genre")
    hit_rate["hit_rate"] = hit_rate["is_hit"] / hit_rate["count"]

    fig_hit_rate = px.bar(
        hit_rate,
        x="genre",
        y="hit_rate",
        title="Proporción de HITS por Género",
        color="hit_rate",
        color_continuous_scale="mint"
    )
    fig_hit_rate.update_yaxes(tickformat="0.0%")
    st.plotly_chart(fig_hit_rate, use_container_width=True)


# 3 Radar Chart por género (NORMALIZADO)

with st.container(border=True):
    st.subheader("🧭 Radar Chart por Género (Normalizado)")

    # Atributos numéricos que usaremos
    radar_columns = [
        "danceability", "energy", "valence",
        "liveness", "acousticness", "instrumentalness",
        "speechiness", "tempo"
    ]

    genero_radar = st.selectbox("Selecciona un género para el radar:", generos_ordenados)

    df_gen = df[df["genre"] == genero_radar][radar_columns].mean()

    # Normalización min-max por columna (mejor comparabilidad)
    df_norm = (df[radar_columns] - df[radar_columns].min()) / (df[radar_columns].max() - df[radar_columns].min())
    df_gen_norm = df_norm[df.index.isin(df[df["genre"] == genero_radar].index)].mean()

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=df_gen_norm.values,
        theta=radar_columns,
        fill='toself',
        name=genero_radar,
        line=dict(color="#32F5C8", width=3)
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showgrid=True,
                gridcolor="gray",
                gridwidth=0.5
            )
        ),
        showlegend=False,
        title=f"Perfil Normalizado de Atributos para {genero_radar}",
        height=500
    )

    st.plotly_chart(fig_radar, use_container_width=True)


# 4 Gráficos dinámicos con selector de atributo

with st.container(border=True):
    st.subheader("🎛 Gráficos Dinámicos por Género")

    # --------- Valores por defecto ----------
    default_left = "danceability"
    default_right = "energy"

    # --------- Seleccion de variables  ----------
    if "attr_left" not in st.session_state:
        st.session_state["attr_left"] = default_left

    if "attr_right" not in st.session_state:
        st.session_state["attr_right"] = default_right

    
    # Restablecer valores por defecto
    
    if st.button("🔄 Restablecer valores por defecto"):
        st.session_state["attr_left"] = default_left
        st.session_state["attr_right"] = default_right

    st.write("")  # Espacio visual

    
    # Selección de genero
    
    generos_ordenados = sorted(df["genre"].unique())
    genero_sel = st.selectbox("Selecciona un género:", generos_ordenados, key="graf_dynamic_genero")

    df_g = df[df["genre"] == genero_sel]

    atributos_numericos = [
        "danceability", "energy", "valence", "loudness",
        "tempo", "acousticness", "instrumentalness",
        "speechiness", "liveness", "duration_ms"
    ]

    col1, col2 = st.columns(2)

    
    # Columna Izquierda
    
    with col1:
        attr_left = st.selectbox(
            "Seleccione el atributo (Gráfico 1):",
            sorted(atributos_numericos),
            key="attr_left"
        )

        fig_left = px.histogram(
            df_g,
            x=attr_left,
            nbins=30,
            title=f"{attr_left.capitalize()} — {genero_sel}",
            color_discrete_sequence=["#32F5C8"]
        )
        st.plotly_chart(fig_left, use_container_width=True)

    
    # Columna Derecha
    
    with col2:
        attr_right = st.selectbox(
            "Seleccione el atributo (Gráfico 2):",
            sorted(atributos_numericos),
            key="attr_right"
        )

        fig_right = px.histogram(
            df_g,
            x=attr_right,
            nbins=30,
            title=f"{attr_right.capitalize()} — {genero_sel}",
            color_discrete_sequence=["#7FFFD4"]
        )
        st.plotly_chart(fig_right, use_container_width=True)



# 5 Comparación HIT vs NO HIT (NORMALIZADO)

with st.container(border=True):
    st.subheader("⚔ Comparación de Atributos: HIT vs NO HIT (Normalizado)")

    atributos = [
        "danceability", "energy", "valence", "loudness",
        "tempo", "acousticness", "instrumentalness",
        "speechiness", "liveness"
    ]

    # Normalización min-max global:
    df_norm = df[atributos].copy()
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())

    hit_avg = df_norm[df["is_hit"] == 1].mean()
    nohit_avg = df_norm[df["is_hit"] == 0].mean()

    comp_df = (
        hit_avg.rename("HIT")
        .to_frame()
        .merge(nohit_avg.rename("NO_HIT").to_frame(), left_index=True, right_index=True)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    fig_comp = px.bar(
        comp_df,
        x="feature",
        y=["HIT", "NO_HIT"],
        barmode="group",
        title="Comparación Normalizada de Atributos: HIT vs NO HIT",
        color_discrete_map={
            "HIT": "#32F5C8",
            "NO_HIT":"#FF8A80"
        }
    )

    fig_comp.update_layout(height=500)
    st.plotly_chart(fig_comp, use_container_width=True)


# 6 PIE HIT vs NO HIT

with st.container(border=True):
    st.subheader("🥧 Distribución HIT vs NO-HIT")

    # Reemplaza 0/1 por etiquetas HIt-No Hit
    hit_counts = df["is_hit"].value_counts().reset_index()
    hit_counts.columns = ["class", "count"]

    # Mapeo personalizado
    class_map = {
        0: "No es HIT",
        1: "Es HIT"
    }

    hit_counts["class_name"] = hit_counts["class"].map(class_map)
    hit_counts["label_legend"] = hit_counts.apply(
        lambda row: f"{row['class_name']} ({row['count']})", axis=1
    )

    fig2 = px.pie(
        hit_counts,
        values="count",
        names="label_legend",      # << Usa la leyenda personalizada
        title="Proporción de canciones HIT",
        color="class",             # << seguimos coloreando por clase original
        color_discrete_map={
            0: "#1a1a1a",          # NO HIT
            1: "#32F5C8"           # HIT
        }
    )

    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)



# STATUS

st.success("Página de métricas avanzadas cargada correctamente.")
