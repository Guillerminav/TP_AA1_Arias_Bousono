# Predicción de lluvia en Australia con Docker

Esta carpeta permite ejecutar el modelo elegido en el trabajo (la red neuronal del punto 7) sobre datos meteorológicos nuevos. El contenedor lee un CSV con registros diarios y devuelve, para cada fila, la probabilidad de que llueva al día siguiente y la predicción `Yes` o `No`.

## Contenido

- `Dockerfile`: arma la imagen a partir de `python:3.11-slim`, instala las dependencias y copia el script y el modelo.
- `requirements.txt`: librerías necesarias solo para predecir, con las mismas versiones con las que se entrenó y se guardó el modelo.
- `inferencia.py`: lee el CSV, aplica el mismo preprocesamiento que en el notebook (fecha cíclica, cambio de presión, viento con seno y coseno, coordenadas y región, imputación y escalado) y predice con la red.
- `modelo_lluvia.keras`: red neuronal entrenada (1 capa oculta de 16 neuronas).
- `preprocesamiento.joblib`: objetos calculados con los datos de entrenamiento: el `StandardScaler`, las medianas para imputar, las coordenadas y la región de cada ciudad, el orden de las columnas y el umbral de decisión.
- `datos/input.csv`: archivo de ejemplo con las 49 ciudades entre el 18 y el 24 de junio de 2017.

## 1. Construir la imagen

Con Docker Desktop abierto, desde esta carpeta (`docker/`):

```bash
docker build -t lluvia-australia-mlops .
```

## 2. Ejecutar el contenedor

El contenedor lee los datos desde la carpeta `datos/`, que se comparte con la opción `-v`. Desde la carpeta `docker/`:

PowerShell, Linux o macOS:

```bash
docker run --rm -v "${PWD}/datos:/data" lluvia-australia-mlops
```

CMD de Windows:

```bash
docker run --rm -v "%cd%/datos:/data" lluvia-australia-mlops
```

Por defecto lee `datos/input.csv` y guarda el resultado en `datos/predicciones.csv`. Para usar otros nombres, se agregan al final la ruta de entrada y la de salida, siempre dentro de `/data`:

```bash
docker run --rm -v "${PWD}/datos:/data" lluvia-australia-mlops /data/mi_archivo.csv /data/mis_predicciones.csv
```

## Formato de entrada

Un CSV con las mismas columnas que `weatherAUS_2026C2.csv`: `Date`, `Location`, `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm` y `RainToday`.

- `Date` en formato `AAAA-MM-DD`.
- `RainToday` como `Yes` o `No`.
- Las columnas `RainTomorrow` y `RainfallTomorrow` no hacen falta. Si vienen, el script las descarta.
- Los valores faltantes pueden quedar vacíos: se completan igual que en el entrenamiento.

## Formato de salida

`predicciones.csv` tiene una fila por cada fila de entrada, en el mismo orden:

| Columna | Descripción |
|---|---|
| `Date` | Fecha del registro |
| `Location` | Ciudad |
| `Probabilidad_Lluvia` | Probabilidad de lluvia para el día siguiente, entre 0 y 1 |
| `Prediccion_RainTomorrow` | `Yes` si la probabilidad supera el umbral elegido en validación, `No` si no |

## Limitaciones

- El cambio de presión y las temperaturas faltantes se calculan con el día anterior de la misma ciudad. Si el CSV trae un solo día por ciudad, esos valores se completan con la mediana de entrenamiento.
- El modelo solo conoce las 49 ciudades del dataset original, porque usa sus coordenadas y su región.