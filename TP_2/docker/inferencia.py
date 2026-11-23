import sys
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

RUMBOS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
ANGULO_RUMBO = {rumbo: i * 360 / 16 for i, rumbo in enumerate(RUMBOS)}
PARES_VIENTO = {
    'WindGustDir': 'WindGustSpeed',
    'WindDir9am': 'WindSpeed9am',
    'WindDir3pm': 'WindSpeed3pm',
}
COLS_TEMP = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']


def preprocesar(df_crudo: pd.DataFrame, art: dict) -> pd.DataFrame:
    df = df_crudo.copy()

    # Descartar columnas que no son predictoras si vienen en el CSV
    for col in ['Unnamed: 0', 'RainfallTomorrow', 'RainTomorrow']:
        if col in df.columns:
            df = df.drop(columns=col)

    df['Date'] = pd.to_datetime(df['Date'])
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

    # Mantener índice original para devolver las predicciones en el mismo orden
    df['_idx_original'] = np.arange(len(df))
    df = df.sort_values(['Location', 'Date']).reset_index(drop=True)
    por_ciudad = df.groupby('Location')
    dia_anterior_existe = por_ciudad['Date'].diff().dt.days.eq(1)

    # 1. Época del año (cíclica)
    dia_del_anio = df['Date'].dt.dayofyear
    angulo_fecha = 2 * np.pi * dia_del_anio / 365.25
    df['Doy_sin'] = np.sin(angulo_fecha)
    df['Doy_cos'] = np.cos(angulo_fecha)

    # 2. Cambio de presión respecto al día anterior en la misma ciudad
    presion_ayer = por_ciudad['Pressure3pm'].shift(1).where(dia_anterior_existe)
    df['Pressure3pm_delta'] = df['Pressure3pm'] - presion_ayer

    # 3. Direcciones de viento cíclicas
    for direccion, velocidad in PARES_VIENTO.items():
        grados = df[direccion].map(ANGULO_RUMBO)
        radianes = np.deg2rad(grados)
        hay_calma = df[velocidad].eq(0)
        df[f'{direccion}_sin'] = np.where(hay_calma, 0.0, np.sin(radianes))
        df[f'{direccion}_cos'] = np.where(hay_calma, 0.0, np.cos(radianes))
    df = df.drop(columns=list(PARES_VIENTO))

    # 4. Coordenadas y regiones aprendidas en train
    df = df.join(art['coords'], on='Location')

    # 5. Temperaturas faltantes: marcar e imputar con día anterior en la misma ciudad
    for col in COLS_TEMP:
        df[f'{col}_falta'] = df[col].isna().astype(int)
        temp_ayer = por_ciudad[col].shift(1).where(dia_anterior_existe)
        df[col] = df[col].fillna(temp_ayer)

    # Restaurar el orden original de las filas antes de preparar la matriz X
    df = df.sort_values('_idx_original').reset_index(drop=True)
    X = df.drop(columns=['Date', 'Location', '_idx_original'])

    # 6. One-Hot de Region y alineación exacta de columnas
    X = pd.get_dummies(X, columns=['Region'], drop_first=True, dtype=int)
    X = X.reindex(columns=art['columnas_modelo'], fill_value=0)

    # 7. Indicadores de faltantes e imputación con las medianas de train
    for col in art['cols_indicador']:
        X[f'{col}_falta'] = X[col].isna().astype(int)
    X[art['cols_viento']] = X[art['cols_viento']].fillna(0)
    X[art['cols_mediana']] = X[art['cols_mediana']].fillna(art['medianas'])

    # Reordenar columnas exactamente igual que en el entrenamiento y escalar
    X = X[art['columnas_modelo']]
    X_scaled = art['scaler'].transform(X).astype('float32')
    return X_scaled


def main():
    ruta_entrada = sys.argv[1] if len(sys.argv) > 1 else '/data/input.csv'
    ruta_salida = sys.argv[2] if len(sys.argv) > 2 else '/data/predicciones.csv'

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada en '{ruta_entrada}'. "
            "Montá la carpeta con los datos usando -v en docker run."
        )

    artefactos = joblib.load('preprocesamiento.joblib')
    modelo = keras.models.load_model('modelo_lluvia.keras')

    df_input = pd.read_csv(ruta_entrada)
    X_proc = preprocesar(df_input, artefactos)

    probs = modelo.predict(X_proc, verbose=0).ravel()
    preds = np.where(probs >= artefactos['umbral'], 'Yes', 'No')

    salida = pd.DataFrame({
        'Date': df_input['Date'] if 'Date' in df_input.columns else np.arange(len(df_input)),
        'Location': df_input['Location'] if 'Location' in df_input.columns else 'N/A',
        'Probabilidad_Lluvia': np.round(probs, 4),
        'Prediccion_RainTomorrow': preds,
    })

    os.makedirs(os.path.dirname(ruta_salida) or '.', exist_ok=True)
    salida.to_csv(ruta_salida, index=False)
    print(f"Inferencia completada ({len(salida)} registros). Resultados guardados en: {ruta_salida}")


if __name__ == '__main__':
    main()