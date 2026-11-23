# Despliegue del Modelo de Predicción de Lluvia (Docker)

Esta carpeta contiene los artefactos y la configuración necesaria para ejecutar inferencias con la red neuronal entrenada (`modelo_lluvia.keras`) sobre nuevos registros meteorológicos en formato CSV.

## Estructura de la carpeta

- `Dockerfile`: definición de la imagen basada en `python:3.11-slim`.
- `requirements.txt`: dependencias mínimas requeridas para el preprocesamiento y la inferencia.
- `inferencia.py`: script que carga los artefactos, aplica el pipeline de transformación y genera las predicciones.
- `modelo_lluvia.keras`: red neuronal densa optimizada.
- `preprocesamiento.joblib`: diccionario serializado con el `StandardScaler`, medianas de imputación, coordenadas/regiones, columnas esperadas y el umbral de decisión óptimo.

## 1. Construir la imagen de Docker

Asegurate de tener **Docker Desktop** abierto y ejecutándose. Luego, abrí una terminal ubicada dentro de esta carpeta (`docker/`) y corré:

```bash
docker build -t lluvia-australia-mlops .
```

## 2. Ejecutar el contenedor

Poner el CSV con los datos en una carpeta (por ejemplo datos/input.csv), con las mismas columnas que weatherAUS_2026C2.csv. Las columnas RainTomorrow y RainfallTomorrow no hacen falta. Desde la carpeta docker/:
 
```bash
docker run --rm -v "${PWD}/datos:/data" lluvia-australia-mlops
```

Las predicciones quedan en `datos/predicciones.csv`, con las columnas `Date`, `Location`, `Probabilidad_Lluvia` y `Prediccion_RainTomorrow`.

Para usar otros nombres de archivo:

```bash
docker run --rm -v "${PWD}/datos:/data" lluvia-australia-mlops /data/mi_archivo.csv /data/salida.csv