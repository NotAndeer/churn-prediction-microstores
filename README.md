# Sistema predictivo e interpretable de abandono de clientes para microtiendas online

## Descripción general

Este repositorio contiene el desarrollo de un sistema predictivo e interpretable que estima la probabilidad de abandono (churn) de clientes en microtiendas online. El sistema integra técnicas de aprendizaje automático y explicabilidad (XAI) para generar predicciones comprensibles y recomendaciones accionables orientadas a mejorar la retención de usuarios.

El enfoque prioriza la reproducibilidad, la aplicabilidad práctica para negocios con recursos limitados y la capacidad de traducir explicaciones de modelos en acciones concretas como cupones, notificaciones o mejoras en la experiencia de usuario.

## Características principales

- Generación de un dataset sintético realista para eventos de navegación y transacciones.
- Pipeline automatizado de limpieza, procesamiento y creación de características.
- Modelos predictivos interpretables y de alto rendimiento.
- Técnicas de interpretabilidad (SHAP, LIME, PDP/ALE).
- Módulo de recomendaciones accionables basado en explicaciones.
- Prototipo web con dashboard para visualización de predicciones y sugerencias.
- Código estructurado para facilitar la reproducibilidad y extensión del proyecto.

## Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/NotAndeer/churn-prediction-microstores.git
cd churn-prediction-microstores
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Generar dataset sintético

```bash
python reproduce.py
```

### 5. Entrenar modelos

```bash
python src/train_models.py
```

### 6. Generar explicaciones y recomendaciones

```bash
python src/generate_explanations.py
```

### 7. Ejecutar prototipo web (si aplica)

```bash
python app/app.py
```

## Estructura del proyecto

```
churn-prediction-microstores
│
├── app/                     # Prototipo web / dashboard
│   └── app.py
│
├── src/
│   ├── data_generation/     # Generación de datos sintéticos
│   ├── preprocessing/       # Limpieza y feature engineering
│   ├── models/              # Entrenamiento y evaluación
│   ├── xai/                 # Interpretabilidad (SHAP, LIME)
│   ├── recommendations/     # Módulo de acciones de retención
│   └── utils/               # Funciones auxiliares
│
├── data/
│   ├── raw/                 # Datos sin procesar
│   └── processed/           # Datos listos para modelado
│
├── notebooks/               # Análisis exploratorios
├── reproduce.py             # Script reproducible de dataset
├── requirements.txt         # Dependencias
└── README.md
```

## Estado del proyecto

| Componente                                  | Estado |
|---------------------------------------------|--------|
| Dataset sintético                           | Completo |
| Preprocesamiento y feature engineering      | Completo |
| Modelos base (logística, árbol simple)      | Implementado |
| Modelos avanzados (XGBoost, EBM)            | En progreso |
| Interpretabilidad (SHAP, LIME, PDP)         | En progreso |
| Motor de recomendaciones                    | Pendiente |
| Prototipo web                               | En desarrollo |
| Validación experimental y documentación     | En progreso |

## Tecnologías utilizadas

- Python 3.x
- pandas, NumPy, scikit-learn
- XGBoost / LightGBM
- SHAP, LIME
- Flask o Streamlit (según implementación)
- Faker, SDV (generación de datos sintéticos)

## Cómo contribuir

1. Crear una rama a partir de `main`.
2. Documentar los cambios en código y README.
3. Realizar pruebas locales.
4. Solicitar revisión mediante Pull Request.

Este proyecto está orientado a investigación académica y puede ser extendido para pruebas reales con microtiendas siempre que se garantice la anonimización de datos.

## Licencia

Especificar la licencia seleccionada para el proyecto (recomendado: MIT). Puede añadirse en un archivo independiente `LICENSE`.

## Autoría

**Freddy Anderson Lombana Cárdenas**  
Proyecto de grado en Ingeniería de Software  
Sistema predictivo e interpretable orientado a la retención de clientes en microtiendas online
