# 📘Guía de Configuración y Flujo de Trabajo – Análisis Musical con Python
**Proyecto:** Análisis de Atributos Musicales y Predicción de Popularidad de Canciones  

![Arquitectura](img/arquitectura.png "1. Arquitectura")

![Entregable](img/entregable-final.png "2. Entregable final")

![Plan](img/plan-proyecto-inicial.png "3. Plan del Proyecto")



## 1. Importar las librerías

Las siguientes librerías son las que se utilizan normalmente en el manejo, análisis y visualización de datos.  

```python
# Parte 1
# Manipulación de datos con DataFrames.
import pandas as pd

# Operaciones numéricas y manejo de arrays.
import numpy as np

# Creación de gráficos básicos (líneas, barras, histogramas, etc.).
import matplotlib.pyplot as plt

# Visualización estadística avanzada (mapas de calor, distribuciones).
import seaborn as sns




# Modelado y algoritmos de aprendizaje automático.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Modelos avanzados de boosting.
import xgboost as xgb
import lightgbm as lgb

# Detección y visualización de datos nulos.
import missingno as msno

# Gráficos interactivos para dashboards.
import plotly.express as px
import plotly.graph_objects as go

# Librerías adicionales para entorno Jupyter y manejo interactivo.
from IPython.display import display, HTML
```

**Descripción breve de librerías utilizadas:**
- **pandas**: lectura, limpieza y manipulación de datos tabulares.  
- **numpy**: operaciones matemáticas y manejo de vectores o matrices.  
- **matplotlib**: gráficos de líneas, barras, histogramas, etc.  
- **seaborn**: visualizaciones estadísticas (mapas de calor, distribuciones).  
- **scikit-learn**: modelado y algoritmos de aprendizaje automático.  
- **missingno**: detección y visualización de datos nulos.  
- **plotly**: gráficos interactivos para dashboards.  
- **jupyter**: entorno interactivo para desarrollo y análisis. 
- **xgboost**:	Implementación optimizada del algoritmo de Extreme Gradient Boosting (XGBoost).
- **lightgbm**:	Algoritmo de Gradient Boosting rápido y eficiente desarrollado por Microsoft. 

---

## 2. Configuración del Entorno Virtual

Para aislar las dependencias del proyecto y mantener versiones estables.

```bash
# Paso 1: Crear entorno virtual, v3.13 es la maxima compatible con streamlit
py -3.13 -m venv .venv 

# Paso 2: Activar entorno virtual
.venv\Scripts\activate

#Paso 2a seleccionar interprete de python para el entorno virtual
a) Presionar Ctrl + shift + P
b) Click en Pyhton: Select Interpreter
c) Seleccionar el que tenga el entorno virtual, ej.  Pyhton 3.13.19(.venv) .\.venv\Scripts\python.exe

# Instalar Jupyter  para visualizar resultado de los archivos de jupyter
pip install jupyter ipykernel

# Paso 3: Instalar librerías necesarias y actualizar pip a su nueva version
pip install pandas numpy matplotlib seaborn tabulate

python.exe -m pip install --upgrade pip

pip install scikit-learn missingno plotly streamlit lightgbm xgboost


# Paso 4: Exportar dependencias instaladas, despues se puede usar el comando pip install -r requirements.txt

pip freeze > requirements.txt

# Paso 5: Clonar el repositorio del proyecto
# (Repositorio documentado en 'Github_notes.md')

# Paso 6: Realizar las actividades asignadas

#pip install fastapi "uvicorn[standard]" pydantic
#API
pip install fastapi "uvicorn[standard]" 

#Dashboard
pip install streamlit plotly




#http://127.0.0.1:8000/docs
```

Comentarios:
- `.venv` crea un entorno virtual local.  
- `pip freeze` genera un archivo con versiones exactas de librerías.  
- `requirements.txt` permite replicar el entorno en otro equipo fácilmente.  

---

## 3. Flujo del Proyecto (Flow Project)

Este flujo organiza las etapas principales del análisis y modelado.

1. Carga de los datos (dataset).  
2. Análisis Exploratorio de los Datos (EDA).  
3. Preparación y tratamiento previo de los datos.  
4. Visualización gráfica de los datos.  
5. Generación del modelo de aprendizaje automático.  
6. Entrenamiento del modelo de aprendizaje automático.  
7. Definición del modelo predictivo.  
8. Evaluación del modelo entrenado con datos reservados.  

### Ejemplos de comandos en EDA

```python
# Vista inicial del dataset
data.head()

# Dimensiones del set de datos
print("Tamaño del set de datos:", data.shape)

# Información general del dataset
data.info()

# Conteo de valores nulos
data.isnull().sum()

# Conteo de registros duplicados
data.duplicated().sum()
```

Comentarios:
- `data.head()` muestra las primeras filas para verificar estructura.  
- `data.shape` indica cuántas filas y columnas contiene.  
- `data.info()` ayuda a detectar tipos de datos y nulos.  
- `isnull()` y `duplicated()` permiten identificar problemas de calidad.  

### Limpieza de Datos (duplicados y nulos)

```python
# Identificar registros duplicados
duplicated_rows = data[data.duplicated()]
print(duplicated_rows)

# Eliminar filas duplicadas
print("Tamaño antes:", data.shape)
data.drop_duplicates(inplace=True)
print("Tamaño después:", data.shape)

# Identificar valores nulos en la columna 'Artist'
null_artists = data[data['Artist'].isnull()]
print("\nÍndices con artistas nulos:")
print(null_artists.index.tolist())

# Eliminar filas con artistas nulos
print("Nulos antes:", data['Artist'].isnull().sum())
data.dropna(subset=['Artist'], inplace=True)
print("Nulos después:", data['Artist'].isnull().sum())
```

Comentarios:
- `data.duplicated()` localiza registros repetidos.  
- `drop_duplicates()` elimina duplicados sin crear una nueva copia.  
- `dropna()` elimina filas con valores faltantes en columnas clave.  

---

## 4. Enfoque del Análisis Exploratorio

Durante el EDA, se analizan principalmente:
- Datos nulos.  
- Registros duplicados.  
- Valores vacíos o inconsistentes.  
- Distribuciones estadísticas de cada atributo (por ejemplo, energy, danceability, valence).  

Estos pasos aseguran una base de datos limpia antes del modelado.  

---

# Estructura del Proyecto

```
CASE-STUDY-SPOTIFY/
PRE-TRAINING/
│
├── .streamlit/
│   └── config.toml
│
├── .venv/
│
├── data/
│   ├── raw/
│   │   └── SpotifyFeatures.csv
│   │
│   ├── processed/
│   │   ├── spotify_clean_modeling.csv
│   │   ├── spotify_clean.csv
│   │   ├── X_test.csv
│   │   ├── y_test.csv
│   │   └── y_pred_model_evaluation.csv
│   │
│   └── processed/extra/ (si deseas mantener otras versiones)
│
├── notebooks/
│   ├── 01_loader.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_trainingold.ipynb
│   └── 05_model_evaluation.ipynb
│
├── notes/
│   ├── img/
│   │   └── img-samples-dashboard/
│   │
│   ├── Github_notes.md
│   ├── Json_API_test.json
│   ├── Markdown_info.md
│   ├── Project Flow.md
│   └── Spotify_Dataset_Description.md
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   └── routes.py
│   │   ├── models/
│   │   │   └── model_pipeline.joblib
│   │   ├── utils/
│   │   │   └── preprocess.py
│   │   ├── static/   (si tu API sirviera archivos estáticos)
│   │   └── requirements_api.txt
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── assets/
│   │   │   └── custom.css
│   │   ├── pages/
│   │   │   ├── 1_metrics.py
│   │   │   ├── gauge.py
│   │   │   ├── utils.py
│   │   │   └── __init__.py
│   │   └── app.py
│   │
│   └── __pycache__/   (ignorado en producción)
│
├── .gitignore
│
├── README.md
│
├── Dockerfile
│
├── requirements.txt         (para entorno general)
└── requirementsApp.txt      (para Streamlit)

```

---

## Recomendaciones

- **processed/**: evita modificar los datos originales; guarda aquí los datasets limpios.  
- **results/**: útil para almacenar gráficas, métricas y comparaciones entre modelos.  
- **utils/**: concentra funciones comunes, como carga de datos, limpieza o visualización.  
- **api/**: te servirá cuando implementes el endpoint `/songs/predict_hit`.  
- **tests/**: si piensas escalar el proyecto o evaluarlo académicamente, esto muestra buenas prácticas.  

---


# Modelos exactos recomendados (clasificación “hit / no hit”)

| Tipo                    | Modelos                                                               | Propósito en tu experimento                        |
|--------------------------|-----------------------------------------------------------------------|----------------------------------------------------|
| **Árboles y Ensambles** | RandomForestClassifier, GradientBoostingClassifier, XGBoost, LightGBM | Modelos potentes, capturan relaciones no lineales. |
| **Lineal**              | LogisticRegression                                                    | Baseline interpretable.                            |
| **Distancia**           | KNeighborsClassifier                                                  | Comparativo, sensible al escalado.                 |

---

| Modelo                   | Librería              | Composición           | Cuándo usarlo                                      | Conversión de `genre` |
|---------------------------|----------------------|-----------------------|----------------------------------------------------|------------------------|
| **RandomForestClassifier**     | `sklearn.ensemble`    | Ensemble (árboles)     | Base sólida, robusto sin escalar.                  | `LabelEncoder` |
| **GradientBoostingClassifier** | `sklearn.ensemble`    | Ensemble (boosting)    | Más preciso, controla bien el overfitting.         | `LabelEncoder` |
| **XGBClassifier**              | `xgboost`             | Boosting avanzado      | Precisión alta, rápido.                            | `LabelEncoder` |
| **LGBMClassifier**             | `lightgbm`            | Boosting optimizado    | Muy rápido en datasets grandes.                    | `LabelEncoder` |
| **LogisticRegression**         | `sklearn.linear_model` | Lineal                 | Buen baseline interpretativo.                      | `OneHotEncoder` |
| **KNeighborsClassifier**       | `sklearn.neighbors`   | Distancia              | Comparativo; sensible al escalado.                 | `OneHotEncoder` |




| Modelo               | Ajuste aplicado              | Efecto                                                  |
| -------------------- | ---------------------------- | ------------------------------------------------------- |
| `RandomForest`       | `class_weight='balanced'`    | Aumenta la importancia de los hits (clase minoritaria). |
| `GradientBoosting`   | sin soporte directo          | Se deja igual, o puedes balancear por resampling.       |
| `XGBoost`            | `scale_pos_weight=ratio`     | Corrige el desbalance en la función de pérdida.         |
| `LightGBM`           | `class_weight='balanced'`    | Pondera internamente las clases.                        |
| `LogisticRegression` | `class_weight='balanced'`    | Ajusta los pesos durante la optimización.               |
| `KNeighbors`         | no soporta pesos automáticos | Se mantiene igual.                                      |


Tu dataset tiene solo 4.53 % de canciones “hit”, lo que provoca que los modelos prioricen predecir “no-hit” (clase 0).
Con class_weight='balanced' y scale_pos_weight, cada modelo penaliza más los errores en la clase minoritaria, mejorando recall y F1-score.


### Analisis de Resultado

| Modelo                  | Accuracy | F1-Score (Hit) | ROC AUC | Conclusiones                                                                                                                   |
| ----------------------- | -------- | -------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **LightGBM**            | 0.8937   | 0.4439         | 0.8785  | Mejor desempeño general. Mantiene alto poder de discriminación y el F1 más equilibrado. Ideal para continuar el entrenamiento. |
| **XGBoost**             | 0.8856   | 0.4262         | 0.8740  | Muy competitivo, pero ligeramente inferior a LightGBM en Recall y estabilidad.                                                 |
| **Random Forest**       | 0.8576   | 0.2837         | 0.6986  | Consistente pero sesgado hacia la clase No-Hit.                                                                                |
| **Logistic Regression** | 0.7779   | 0.2646         | 0.8122  | Base lineal razonable, pero limitada para capturar relaciones complejas.                                                       |
| **Gradient Boosting**   | 0.9554   | 0.1321         | 0.5356  | Accuracy inflado; pobre desempeño en detección de Hits.                                                                        |
| **K-Neighbors**         | 0.9520   | 0.0809         | 0.5214  | Alto Accuracy por sesgo hacia No-Hit. Ineficiente para identificar Hits.                                                       |

---

### 🏁 Conclusión final

- **LightGBM** → modelo óptimo para pasar al archivo `04_model_training.ipynb`.  
- **XGBoost** → referencia secundaria para comparar después del ajuste de hiperparámetros.

---

### 📊 Métricas utilizadas

- **accuracy_score** → mide qué proporción de predicciones fueron correctas.  
- **f1_score** → mide el equilibrio entre *precisión* y *recall* (útil si las clases están desbalanceadas).  
- **roc_auc_score** → mide la capacidad del modelo para distinguir entre clases (cuanto más cerca de 1, mejor).


1. Logistic Regression

Modelo lineal.
Sirve como baseline. Rápido, interpretable y muestra qué variables empujan a la probabilidad de ser hit.

2. Random Forest

Ensamble de muchos árboles de decisión.
Robusto, maneja no-linealidades y detecta interacciones entre features automáticamente.

3. Gradient Boosting (GBM clásico de sklearn)

Construye árboles de manera secuencial, corrigiendo errores del anterior.
Mejor rendimiento que RandomForest pero más lento.

4. XGBoost

Implementación optimizada y más poderosa de boosting.
Alta precisión, muy usado en competencias de Kaggle. Excelente con datasets tabulares.

5. LightGBM

Boosting muy rápido desarrollado por Microsoft.
Funciona excelente con grandes volúmenes (como tu dataset de 230k filas).
Suele superar a XGBoost en velocidad con rendimiento similar o mejor.

Métricas que se van a comparar

Debido al dataset desbalanceado (4.6% hits), no sirve usar solo accuracy.
Por eso se evalúan 3 métricas clave:

1. Accuracy

Porcentaje de predicciones correctas.
No es muy útil con desbalance (un modelo que prediga “todo es no-hit” ya logra 95%).

2. F1-score

Promedio entre precision y recall para la clase positiva (hit).
Es la métrica crítica cuando la clase “hit” es muy minoritaria.
Evalúa qué tan bien detecta hits sin generar demasiados falsos positivos.

3. ROC-AUC

Mide la capacidad del modelo de separar ambas clases.
No depende del umbral 0.5.
Valores:

0.5 = aleatorio

1.0 = perfecto
Un buen modelo suele estar > 0.80

En una frase:

Entrenaremos 5 algoritmos (lineales, árboles y boostings) y los compararemos usando métricas robustas frente al desbalance (F1 y AUC) para seleccionar el mejor modelo que predice si una canción puede ser un hit.

Estructura Final SRC
API FastAPI → carpeta src/api/

Modelo entrenado → carpeta src/api/models

Dashboard Streamlit → carpeta src/dashboard/


Publicacion 
                 +------------------------+
                 |   Streamlit Cloud      |
                 |   (Dashboard UI)       |
                 |   https://...app       |
                 +-----------+------------+
                             |
                             |  HTTPS (POST/JSON)
                             v
         +---------------------------------------------+
         |  Railway.app (API FastAPI + Modelo ML)      |
         |  https://<project>.railway.app/predict_hit   |
         +---------------------------------------------+
Dashboard → Streamlit Cloud (sin Docker)
API FastAPI → Railway (con Docker obligatorio)