"""
Módulo: main.py

Responsabilidad:
----------------
Punto de entrada auxiliar para el proyecto en modo desarrollo.

Permite, por ejemplo:
- arrancar la API FastAPI con uvicorn, o
- imprimir información de diagnóstico del entorno.

Uso típico durante desarrollo:
------------------------------
    python main.py api

    python main.py info
"""

from __future__ import annotations

import sys


def run_api():
    """
    Arranca la API utilizando uvicorn.
    """
    import uvicorn

    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def show_info():
    """
    Muestra información básica de ayuda.
    """
    print("Comandos disponibles:")
    print("  python main.py api    → Arranca la API FastAPI")
    print("  python main.py info   → Muestra esta ayuda")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_info()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "api":
        run_api()
    else:
        show_info()
