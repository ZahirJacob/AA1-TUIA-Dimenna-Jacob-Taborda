import joblib
import logging
from sys import stdout
import warnings
import pandas as pd
from pycaret.classification import load_model, predict_model
from preprocesamiento import preprocesar_datos
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


# Cargar input crudo
df_input = pd.read_csv("/files/input.csv")
logger.info("loaded input")

# Aplicar el mismo preprocesamiento usado en entrenamiento
df_procesado = preprocesar_datos(df_input)
logger.info("preprocessed input")

# Cargar modelo entrenado con PyCaret
modelo = load_model("pipeline")
logger.info("loaded pipeline")

# Hacer predicciones
predicciones = predict_model(modelo, data=df_procesado)
logger.info("made predictions")

salida = pd.DataFrame()

salida["prediccion"] = predicciones["prediction_label"].map({
    0: "No llueve",
    1: "Llueve"
})


if "prediction_score" in predicciones.columns:
    salida["Probabilidad"] = predicciones["prediction_score"]

salida.to_csv("/files/output.csv", index=False)
logger.info("saved output")