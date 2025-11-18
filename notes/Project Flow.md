# 📘Configuración y Flujo de Trabajo – Análisis Musical con Python –

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
---
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

## 2. Configuración del Entorno Virtual y Desarrollo del Proyecto

Para aislar las dependencias del proyecto y mantener versiones estables.

```bash
# Paso 1: Crear entorno virtual, v3.13 es la maxima compatible con streamlit
py -3.13 -m venv .venv 

# Paso 2: Activar entorno virtual
.venv\Scripts\activate

#Paso 2a seleccionar interprete de python para el entorno virtual
a) Presionar Ctrl + shift + P
b) Click en Pyhton: Select Interpreter
c) Seleccionar el que tenga el entorno virtual. Pyhton 3.13.19(.venv) .\.venv\Scripts\python.exe

# Instalar Jupyter  para visualizar resultado de los archivos de jupyter
pip install jupyter ipykernel

# Paso 3: Instalar librerías necesarias y actualizar pip a su nueva version
pip install pandas numpy matplotlib seaborn tabulate

python.exe -m pip install --upgrade pip

pip install scikit-learn missingno plotly streamlit lightgbm xgboost

# Paso 4: Exportar dependencias instaladas, despues se puede usar el comando pip install -r requirements.txt

pip freeze > requirements.txt

# Paso 5: Clonar el repositorio del proyecto
 (Repositorio documentado en 'Github_notes.md')

# Paso 6: Realizar las actividades asignadas y creacion del modelo.

# Paso 7: Creacion del API con FastAPI
pip install fastapi "uvicorn[standard]" pydantic

# Paso 8: Pruebas de la API en entorno virtual
# http://127.0.0.1:8000/docs
uvicorn src.api.main:app --reload

# Paso 9: Creacion de requirements_api.txt para que docker use los requisit
pip freeze > requirements_api.txt

# Paso 9: Creacion y prueba del Dashboard con Streamlit
pip install streamlit plotly
streamlit run app.py

# Paso 10: Creacion del archivo requirements.txt para la publicación del Dashboard. 
pip freeze > requirements_api.txt

```

Comentarios:
- `.venv` crea un entorno virtual local.  
- `pip freeze` genera un archivo con versiones exactas de librerías.(crear una para Docker, una para API y una general)  
- `requirementsApp.txt` permite replicar el entorno en otro equipo fácilmente.  
- `requirements_api.txt` permite replicar el entorno con Docker.  
- `requirements.txt` permite realizar la publicación en Streamlite Cloud.  

---

## 3. Flujo del Proyecto (Flow Project)

Este flujo organiza las etapas principales del análisis y modelado.

1. Carga de los datos (dataset).  
2. Análisis Exploratorio de los Datos (EDA).  
3. Preparación y tratamiento previo de los datos.  
4. Visualización gráfica de los datos.  
5. Generación de los modelos de aprendizaje automático.  
6. Entrenamiento del los modelos de aprendizaje automático.  
7. Definición final del modelo predictivo y entrenamiento.  
8. Evaluación del modelo entrenado con datos reservados.
9. Creacion del modelo joblib
10. Creacion de la API, mediante Fast API
11. Prueba local y publicación
12. Creación y prueba del Dashboard usando Streamlit
13. Publicación del Dashboard

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
### Tipos de valores faltantes en Python y pandas

#### **NaN — Not a Number**
Valor especial del tipo `float`, proveniente de NumPy, utilizado para representar **datos faltantes numéricos**.

**Características:**
- Tipo: `float`
- `NaN != NaN`
- Propaga en operaciones matemáticas
- Se usa en columnas numéricas

**Ejemplo:**
```python
import numpy as np

x = np.nan
print(type(x))      # float
print(x == x)       # False
```

---

#### **None — Valor nulo en Python (similar a NULL)**
Representa ausencia de valor en Python.

**Características:**
- Tipo: `NoneType`
- No se puede usar en operaciones matemáticas
- Común en columnas tipo `object` (texto)

**Ejemplo:**
```python
x = None
print(type(x))      # NoneType
```

---

## Uso interno en Pandas

| Tipo de columna | Valor faltante usado |
|----------------|----------------------|
| Numéricas      | `NaN`                |
| Strings/Objetos | `None` o `pd.NA`     |
| Tipos extendidos (Int64, boolean, string) | `pd.NA` |

---
## 4. Estructura General del Proyecto, EDA, API, Dashboard

**Estructura del Proyecto**

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
│   │   ├── static/   (Cuando la API sirve archivos estáticos, ej. Icono)
│   │   └── requirements_api.txt      (Usando para crear imagen de Docker)
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
├── requirements.txt         (para Streamlit)
└── requirementsApp.txt      (para entorno general)

```

---
## 5. Enfoque del Análisis Exploratorio EDA

Durante el EDA, se analizan principalmente:
- Datos nulos.  
- Registros duplicados.  
- Valores vacíos o inconsistentes.  
- Distribuciones estadísticas de cada atributo (por ejemplo, energy, danceability, valence).  

Estos pasos aseguran una base de datos limpia antes del modelado.  

---
## 5.1. Mejores Prácticas

- **processed/**: evita modificar los datos originales; lugar donde se guardan los datasets limpios.  
- **utils/**: concentra funciones comunes, como carga de datos, limpieza o visualización.  
- **api/**: Core para la creación del API con todos sus endpoint(s) `/songs/predict_hit`.  
- **dashboard/**: Core para la creación de la visualización Front-End de la app.  
---


# 5.2. Modelos que podrian usarse para (clasificación “hit / no hit”)

| Tipo                    | Modelos                                                               | Propósito en tu experimento                        |
|--------------------------|-----------------------------------------------------------------------|----------------------------------------------------|
| **Árboles y Ensambles** | RandomForestClassifier, GradientBoostingClassifier, XGBoost, LightGBM | Modelos potentes, capturan relaciones no lineales. |
| **Lineal**              | LogisticRegression                                                    | Baseline interpretable.                            |
| **Distancia**           | KNeighborsClassifier                                                  | Comparativo, sensible al escalado.                 |

---

| Modelo                   | Librería              | Composición           | Cuándo usarlo                                      | Conversión de `genre` |
|---------------------------|----------------------|-----------------------|----------------------------------------------------|------------------------|
| **RandomForestClassifier**     | `sklearn.ensemble`    | Ensemble (árboles)     | Base sólida, robusto sin escalar.                  | `LabelEncoder` |
| **GradientBoostingClassifier** | `sklearn.ensemble`    | Ensemble (boosting)    | Más preciso, controla bien el overfitting (sobreajuste extremo).         | `LabelEncoder` |
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


El dataset tiene solo 4.53 % de canciones “hit”, lo que provoca que los modelos prioricen predecir “no-hit” (clase 0).
Con class_weight='balanced' y scale_pos_weight, cada modelo penaliza más los errores en la clase minoritaria, mejorando recall y F1-score.


### 5.3. Análisis de Resultados

| Modelo                  | Accuracy | F1-Score (Hit) | ROC AUC | Conclusiones                                                                                   |
|-------------------------|----------|----------------|---------|------------------------------------------------------------------------------------------------|
| **LightGBM**            | 0.929552 | 0.531505       | 0.891965 | Mejor desempeño del batch. Alto poder de discriminación y mejor F1 en detección de Hits.        |
| **XGBoost**             | 0.926609 | 0.523438       | 0.893899 | Muy sólido y cercano a LightGBM. Excelente AUC y buen equilibrio entre precisión y recall.      |
| **Logistic Regression** | 0.777828 | 0.263829       | 0.810842 | Interpretación sencilla, pero limitada para capturar patrones complejos del dataset.            |
| **Random Forest**       | 0.755720 | 0.244518       | 0.796417 | Modelo estable pero sesgado hacia la clase mayoritaria (No-Hit).                                |
| **Gradient Boosting**   | 0.955892 | 0.143513       | 0.538895 | Accuracy engañosamente alto; falla en identificar Hits.                                         |
| **K-Neighbors**         | 0.953378 | 0.020758       | 0.505191 | Muy mal F1 para Hits. Predomina completamente la clase No-Hit.                                  |

---

### 🏁 Conclusión final

- **LightGBM** → es el modelo óptimo para el step 4 `04_model_training.ipynb`.  
- **XGBoost** → referencia secundaria para comparar después del ajuste de hiperparámetros.

---

### 5.4. 📊 Métricas utilizadas
Debido al dataset desbalanceado (4.6% hits), no es suficiente usar solo accuracy.
Por eso se evaluarán 3 métricas clave y una adicional al validar el modelo. Matriz de confusión.

- **accuracy_score** → mide qué proporción de predicciones fueron correctas.  
- **f1_score** → mide el equilibrio entre *precisión* y *recall* (útil si las clases están desbalanceadas).  
- **roc_auc_score** → mide la capacidad del modelo para distinguir entre clases (cuanto más cerca de 1, mejor).

**Notes**
1. Accuracy
No es muy útil con un desbalance (un modelo que prediga “todo es no-hit” ya logra 95%).

2. F1-score
Es la métrica crítica cuando la clase “hit” es muy minoritaria.
Evalúa qué tan bien detecta hits sin generar demasiados falsos positivos.

3. ROC-AUC
No depende del umbral por defecto de 0.5.
Valores:
0.5 = aleatorio
1.0 = perfecto (un buen modelo suele estar > 0.80).


### 5.5 🕵️ Modelos a evaluar 

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

### 5.6 Resumen EDA
Entrenaremos 5 algoritmos (lineales, árboles y boostings) y los compararemos usando métricas robustas frente al desbalance (F1 y AUC) para seleccionar el mejor modelo que predice si una canción puede ser un hit.


### 6. Resumen Final Entregables

**Estructura Final SRC**

- API FastAPI → carpeta src/api/
- Modelo entrenado → carpeta src/api/models
- Dashboard Streamlit → carpeta src/dashboard/


**Publicación** 

```
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
```
