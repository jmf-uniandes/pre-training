import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 1. Definición del CSS incrustado para el efecto NEÓN ---
NEON_CSS = """
<style>
/* =========================================================
   Estilos de Streamlit para asegurar un fondo oscuro
   ========================================================= 
*/
body {
    background-color: #121212; 
}

/* 2. Aplicar Brillo NEÓN al Número Central, Título y % */
/* Plotly renderiza el texto en SVG. Apuntamos a los elementos <text> generados. */
.main-svg text {
    fill: #00F0FF !important; /* Forzar el color base neón (cyan) */
    font-weight: bold !important;
}

/* Brillo NEÓN para el texto de valor y título */
.main-svg .g-gtitle .g-text, 
.main-svg .infolayer .g-gtitle .g-text,
.main-svg .g-number .g-text {
    text-shadow: 
        0 0 5px rgba(0, 240, 255, 0.5), 
        0 0 10px rgba(0, 240, 255, 0.8), 
        0 0 15px #00F0FF; 
}

/* Reducir un poco el brillo del subtítulo para que sea más sutil */
.main-svg .infolayer .g-title {
    text-shadow: 
        0 0 3px rgba(0, 240, 255, 0.4),
        0 0 7px #00F0FF;
    font-size: 18px !important;
    letter-spacing: 1px;
    font-weight: normal !important;
}

</style>
"""

def inject_css(css_code):
    """Inyecta el CSS global en el DOM de Streamlit."""
    st.markdown(css_code, unsafe_allow_html=True)


## 🎯 Función para crear el Gauge de Plotly 
def create_gauge_chart(value, title_text):
    """Crea y devuelve un gráfico de gauge de Plotly con estilo futurista (Circular)."""
    
    # Colores base
    NEON_COLOR_ARC = "#00FFC2"  # Color para el arco progresivo (verde/cyan)
    BG_ARC_COLOR = "#1A1A1A"    # Gris oscuro para el fondo del arco
    NEON_COLOR_TEXT = "#00F0FF" # Color para el texto (cyan puro, usado en CSS)

    value = np.clip(value, 0, 100) 
    
    fig = go.Figure(go.Indicator(
        # CORRECCIÓN DE ERROR: Eliminamos '+title' de mode
        mode="gauge+number", 
        value=value,
        title={'text': title_text},
        
        # Configuramos el número central para incluir el '%' y color base
        number={'suffix': "%", 'font': {'color': NEON_COLOR_TEXT, 'size': 50}}, 
        
        # DOMAIN: Fijo en [0, 1] para forzar la forma.
        domain={'x': [0, 1], 'y': [0, 1]}, 
        gauge={
            'bar': {'color': NEON_COLOR_ARC, 'thickness': 0.28},
            'bgcolor': BG_ARC_COLOR,
            
            # AXIS: Ocultamos los números, solo mostramos los ticks sutiles.
            'axis': {
                'range': [0, 100], 
                'tickwidth': 1, 
                'tickcolor': "rgba(255, 255, 255, 0.2)",
                'showticklabels': False, 
            },
            
            # STEPS: El rango completo en gris oscuro es el fondo del medidor.
            'steps': [
                {'range': [0, 100], 'color': BG_ARC_COLOR} 
            ],
        }
    ))

    # LAYOUT: Establecer el tamaño fijo y la transparencia.
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        
        # Tamaño fijo para forzar la forma circular.
        height=350, 
        width=350,  
        
        # Elimina las interacciones de Plotly que no queremos ver
        hovermode=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# =========================================================
# 3. LÓGICA PRINCIPAL DE STREAMLIT (Llamada a las funciones)
# =========================================================

# --- Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Gauge Demo Incrustado",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyectar el CSS antes que cualquier otro contenido
inject_css(NEON_CSS)

st.title("🎯 Demo de Gauge Estilo Futurista (CSS Incrustado)")

# Slider para control interactivo
probabilidad_hit = st.slider(
    "Selecciona la Probabilidad de Hit (%)", 
    0, 100, 87, 
    key='slider_prob'
)

# Crear el gráfico
gauge_fig = create_gauge_chart(
    value=probabilidad_hit,
    title_text="PROBABILIDAD DE HIT"
)

# Mostrar el gráfico en Streamlit.
with st.container():
    # Centrar el elemento Plotly en el contenedor
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(gauge_fig, use_container_width=False)

st.divider()
st.info(f"El valor actual de la probabilidad es: **{probabilidad_hit}%**")