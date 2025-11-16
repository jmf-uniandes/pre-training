import plotly.graph_objects as go
import numpy as np

# Colores neón y fondo
NEON_COLOR_ARC = "#00FFC2"
NEON_COLOR_TEXT = "#00F0FF"
BG_ARC_COLOR = "#1A1A1A"


def create_gauge_chart(value, title_text="PROBABILIDAD DE HIT"):
    """
    Gauge de estilo moderno para Streamlit (Plotly).
    - value: Número entre 0 y 100
    - title_text: Texto superior del gauge
    """

    # Seguridad: el valor debe estar entre 0–100
    value = np.clip(value, 0, 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,

        # =======================
        # TÍTULO DEL GAUGE
        # =======================
        title={
            "text": f"<b>{title_text}</b>",
            "font": {
                "size": 20,
                "color": NEON_COLOR_TEXT,
                "family": "Orbitron"
            }
        },

        # =======================
        # NÚMERO CENTRAL
        # =======================
        number={
            'suffix': "%",
            'font': {
                'color': NEON_COLOR_TEXT,
                'size': 50,
                'family': "Orbitron"
            }
        },

        # =======================
        # DOMINIO
        # =======================
        domain={'x': [0, 1], 'y': [0, 1]},

        # =======================
        # DEFINICIÓN DEL GAUGE
        # =======================
        gauge={
            'shape': "angular",
            'bar': {
                'color': NEON_COLOR_ARC,
                'thickness': 0.25
            },

            'bgcolor': BG_ARC_COLOR,

            # Eje semicircular
            'axis': {
                'range': [0, 100],
                'tickwidth': 0,
                'showticklabels': False,
            },

            # Banda de fondo
            'steps': [
                {'range': [0, 100], 'color': BG_ARC_COLOR}
            ],
        }
    ))

    # =======================
    # LAYOUT GENERAL
    # =======================
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        height=None,
        width=None,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    # Mantener figura perfectamente cuadrada
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
