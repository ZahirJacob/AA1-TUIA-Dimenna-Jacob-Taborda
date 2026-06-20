import joblib
import numpy as np
import pandas as pd


def asignar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Summer"
    elif mes in [3, 4, 5]:
        return "Autumn"
    elif mes in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"


def obtener_valor_por_region_season(tabla, clave, variable, valor_global):
    """
    Busca un valor en una tabla por Region y Season.
    Si no encuentra la combinación, usa el valor global.
    """
    try:
        valor = tabla.loc[clave, variable]
    except Exception:
        valor = valor_global

    if pd.isna(valor):
        valor = valor_global

    return valor


def preprocesar_datos(df, artifacts_path="preprocessing_training_data.pkl"):
    """"""
    artifacts = joblib.load(artifacts_path)

    df = df.copy()

    # Eliminar columnas que no deben entrar al modelo
    df = df.drop(
        columns=["Unnamed: 0", "RainTomorrow", "RainfallTomorrow"],
        errors="ignore"
    )

    # Date -> Season
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["Season"] = df["Month"].apply(asignar_estacion)
        df = df.drop(columns=["Date", "Month"], errors="ignore")

    if "Season" not in df.columns:
        df["Season"] = artifacts["season_default"]

    df["Season"] = df["Season"].fillna(artifacts["season_default"])

    # Location -> Region
    if "Location" in df.columns:
        location_region_map = artifacts["location_region_map"]
        df["Region"] = df["Location"].map(location_region_map)
        df = df.drop(columns=["Location"], errors="ignore")

    if "Region" not in df.columns:
        df["Region"] = artifacts["region_default"]

    df["Region"] = df["Region"].fillna(artifacts["region_default"])

    # Imputación por mediana Region + Season
    variables_mediana = artifacts["variables_mediana"]
    medianas_region_season = artifacts["medianas_region_season"]
    medianas_globales = artifacts["medianas_globales"]

    for col in variables_mediana:
        if col not in df.columns:
            df[col] = np.nan

    for variable in variables_mediana:
        filas_nulas = df[df[variable].isna()].index

        for idx in filas_nulas:
            region = df.loc[idx, "Region"]
            season = df.loc[idx, "Season"]
            clave = (region, season)

            valor = obtener_valor_por_region_season(
                tabla=medianas_region_season,
                clave=clave,
                variable=variable,
                valor_global=medianas_globales[variable]
            )

            df.loc[idx, variable] = valor

    # Escalado de columnas numéricas con el scaler entrenado
    columnas_numericas = artifacts["columnas_numericas"]
    scaler = artifacts["scaler"]

    for col in columnas_numericas:
        if col not in df.columns:
            df[col] = np.nan

    df[columnas_numericas] = scaler.transform(df[columnas_numericas])

    # Imputación KNN por Region + Season
    variables_knn = artifacts["variables_knn"]
    imputers_knn = artifacts["imputers_knn"]
    global_knn_imputer = artifacts["global_knn_imputer"]

    for col in variables_knn:
        if col not in df.columns:
            df[col] = np.nan

    for (region, season), grupo in df.groupby(["Region", "Season"]):
        clave = (region, season)

        if clave in imputers_knn:
            imputer = imputers_knn[clave]
        else:
            imputer = global_knn_imputer

        df.loc[grupo.index, variables_knn] = imputer.transform(
            grupo[variables_knn]
        )

    # Imputación categórica por moda Region + Season
    variables_categoricas_imputar = artifacts["variables_categoricas_imputar"]
    modas_region_season = artifacts["modas_region_season"]
    modas_globales = artifacts["modas_globales"]

    for col in variables_categoricas_imputar:
        if col not in df.columns:
            df[col] = np.nan

    for variable in variables_categoricas_imputar:
        filas_nulas = df[df[variable].isna()].index

        for idx in filas_nulas:
            region = df.loc[idx, "Region"]
            season = df.loc[idx, "Season"]
            clave = (region, season)

            valor = obtener_valor_por_region_season(
                tabla=modas_region_season,
                clave=clave,
                variable=variable,
                valor_global=modas_globales[variable]
            )

            df.loc[idx, variable] = valor

    # Alinear columnas igual que en entrenamiento
    columnas_modelo = artifacts["columnas_modelo"]

    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = np.nan

    df = df[columnas_modelo]

    return df