# Plan de Proyecto Inicial — Seminario de Analítica con Python

## 1. Título del Proyecto y Miembros

**Título del Proyecto:**  
Análisis de Atributos Musicales y Predicción de Popularidad de Canciones.

**Miembros del Equipo:**  
- José Miguel Figueroa Chilan  
- Edison Fernando Gonzalez Alberca  
- Carlos Bernardo Pintag Quispe

---

## 2. Definición del Problema

La industria musical (sellos discográficos, artistas, managers, curadores de playlists y plataformas de streaming) necesita anticipar qué canciones tienen potencial de convertirse en éxitos para optimizar inversiones en promoción, marketing y posicionamiento. Sin una analítica objetiva basada en datos de audio, la toma de decisiones se apoya principalmente en la intuición, lo que incrementa el riesgo de invertir en canciones con baja probabilidad de éxito y desaprovechar temas con alto potencial.

### Objetivo Principal
Construir un sistema de analítica que, a partir de los atributos de una canción de Spotify, clasifique automáticamente si tiene perfil de “hit” o “no hit” mediante un modelo de Machine Learning, exponiendo la predicción a través de una API y un dashboard interactivo denominado **“El Buscador de Hits”**.

---

## 3. Columnas Clave del Dataset

1. **popularity**  
   Variable núcleo del problema; permite definir la variable objetivo binaria *is_hit*.

2. **genre**  
   Diferentes géneros presentan patrones distintos de popularidad.

3. **danceability**  
   Canciones bailables suelen tener más probabilidad de éxito.

4. **energy**  
   La intensidad sonora se asocia a hits de radio y playlists populares.

5. **valence**  
   Indica cuán “feliz” o positiva es la canción.

6. **tempo**  
   Asociado al ritmo, importante para entender patrones de hits.

7. **acousticness**  
   Ayuda a distinguir producciones acústicas vs electrónicas.

8. **duration_ms**  
   La duración influye en tendencias modernas del consumo musical.

---

## 4. Responsabilidades de la Arquitectura

### Pipeline
- Carga y limpieza del dataset (duplicados, nulos, tipos).
- Creación de *is_hit* y features derivadas.
- División train/test, manejo de desbalance y entrenamiento de modelos.
- Selección del mejor modelo y guardado del pipeline final.

### API
- Endpoint `/songs/predict_hit` que reciba atributos y devuelva probabilidad y clasificación.
- Carga del modelo y lógica de predicción.
- Documentación automática (Swagger).
- Endpoints auxiliares opcionales.

### Dashboard
- UI interactiva para enviar atributos a la API.
- Visualización del resultado mediante gauge y mensajes interpretativos.
- Sección de EDA con gráficos explicativos.
- Diseño consistente y moderno.

---

## 5. Primeras Tareas Técnicas (Sesión 5)

1. **Crear repositorio en GitHub**  
   Responsable: *José Miguel Figueroa Chilan*

2. **Redactar README inicial**  
   Responsable: *Carlos Bernardo Pintag Quispe*

3. **Crear script de carga de datos**  
   Responsable: *Edison Gonzalez Alberca*

---

## 6. Dudas y Riesgos Identificados

- Desbalance de clases (pocos hits vs muchos no hits).  
- Elección del umbral de decisión para clasificar hit/no hit.  
- Generalización del modelo a géneros minoritarios o música futura.  
- Tiempos de entrenamiento por el tamaño del dataset.  
- Estabilidad en la comunicación API–Dashboard (CORS, red).  
- Interpretabilidad del modelo para la defensa del proyecto.

