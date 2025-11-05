
# 📘 Git Commit & Branching Specification  
**Proyecto:** Análisis de Atributos Musicales y Predicción de Popularidad de Canciones  

---

## Convención de Commits

Usa el formato estándar de *Conventional Commits* para mantener un historial claro, legible y trazable:

### Tipos permitidos

| Tipo | Descripción | Ejemplo |
|------|--------------|----------|
| **feat** | Nueva funcionalidad o módulo | `feat(api): add /songs/predict_hit endpoint` |
| **fix** | Corrección de bug o comportamiento incorrecto | `fix(api): correct JSON key error in predict_hit response` |
| **refactor** | Mejora de código sin cambiar funcionalidad | `refactor(eda): modularize correlation heatmap function` |
| **docs** | Cambios en documentación o comentarios | `docs: update README with setup instructions` |
| **style** | Cambios estéticos o de formato | `style: reformat imports and indentation` |
| **test** | Creación o mejora de pruebas unitarias | `test(model): add unit tests for hit predictor` |
| **data** | Limpieza o modificación del dataset | `data: create binary target column hit from popularity > 75` |
| **model** | Cambios en modelos de machine learning | `model: train RandomForest with tuned hyperparameters` |
| **api** | Cambios en endpoints o estructura del backend | `api: add Swagger documentation for predict_hit` |
| **ui** | Cambios en interfaz o dashboard interactivo | `ui: add real-time probability chart in dashboard` |
| **chore** | Mantenimiento, configuración o tareas auxiliares | `chore: add .env.example and update .gitignore` |

---

## Ejemplos de Commits

```bash
feat(api): add /songs/predict_hit endpoint
model: train RandomForest with tuned hyperparameters
data: create binary target column hit from popularity > 75
refactor(eda): modularize correlation heatmap function
fix(api): correct JSON key error in predict_hit response
docs: update README with setup and API usage instructions
ui: add real-time probability chart in dashboard
chore: add .env.example and update .gitignore

```

### Ramas Sugeridas
main           -> versión estable
dev            -> integración de nuevas funciones
feature/...    -> desarrollo de funciones específicas
fix/...        -> corrección de errores
model/...      -> experimentos de ML

main
└── dev
    ├── data/ingestion       # Carga del dataset original
    ├── data/eda             # Análisis exploratorio de datos
    ├── data/preprocessing   # Limpieza y normalización
    ├── viz/analysis         # Visualización gráfica y correlaciones
    ├── model/build          # Creación de arquitectura base del modelo
    ├── model/training       # Entrenamiento y ajuste de hiperparámetros
    ├── model/predictive     # Generación del modelo final y API
    └── model/evaluation     # Evaluación con datos reservados



## Flujo de Trabajo Recomendado

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/usuario/proyecto.git
   ```

2. **Configurar tu identidad de usuario**
   ```bash
   git config --global user.name "Jose Miguel"
   git config --global user.email "josefc79@uniandes.edu.ec"
   ```

3. **Crear una nueva rama**
   ```bash
   git checkout -b <nombre-de-la-nueva-rama>
   ```

4. **Agregar cambios**
   ```bash
   git add archivo.ext   # o para agregar todos los cambios
   git add .
   ```

5. **Realizar un commit**
   ```bash
   git commit -m "feat(api): add /songs/predict_hit endpoint"
   ```

6. **Subir cambios al repositorio remoto**  
   Este paso envía tu nueva rama y sus commits al servidor remoto.

   ```bash
   git push -u origin <nombre-de-la-nueva-rama>
   # git push        → envía los commits locales al repositorio remoto.
   # -u              → establece un enlace entre la rama local y la rama remota (upstream).
   # origin          → nombre del repositorio remoto por defecto.
   # <nombre-de-la-nueva-rama> → nombre de la rama que estás subiendo (ej. feature/api-endpoint).


7. **Fusionar cambios aprobados**
   ```bash
    git checkout main
    git pull origin main
    git merge <nombre-de-la-nueva-rama>
    git push origin main

    git checkout main 
    # Cambia a la rama principal (main) para preparar la fusión.

    git pull origin main  
    # Actualiza la rama main con la última versión del repositorio remoto
    # (descarga nuevos commits de otros colaboradores, si existen).

    git merge <nombre-de-la-nueva-rama>  
    # Combina el contenido de la rama especificada dentro de main.
    # Si no hay conflictos, los cambios quedan integrados.

    git push origin main   # Sube la versión actualizada de main al repositorio remoto,
    # reflejando la integración completada.
   ```

8. **Crear y gestionar un Pull Request (PR)**  
   Un *Pull Request* (también llamado *Merge Request*) se utiliza para revisar, discutir y aprobar los cambios antes de fusionarlos con la rama principal.

### 🔹 Paso a paso para crear un Pull Request

1. **Sube tu rama al repositorio remoto**
   ```bash
   git push -u origin <nombre-de-la-nueva-rama>
   # Sube la rama local al servidor remoto para poder crear el PR.

---

## 🧭 Buenas prácticas

- Commits **frecuentes y pequeños** (1 cambio lógico por commit).  
- Nombres de ramas **en minúsculas y separados por guiones** (`feature/api-endpoint`).  
- Actualizar `main` antes de cada merge.  
- Mantener descripciones claras en los *Pull Requests*.  
- Documentar cambios relevantes en `CHANGELOG.md` o `README.md`.

---
