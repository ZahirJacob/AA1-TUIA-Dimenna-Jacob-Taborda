# Inferencia - Predicción de lluvia en Australia

El contenedor predice si va a llover al dia siguiente a partir de datos
medidos el dia anterior, implementado con PyCaret.

## Build
Desde `docker/` ejecutar:

```bash
docker build -t prediccion-lluvia .
```

## Run
```bash
docker run -v ${PWD}/files:/files prediccion-lluvia
```

Salida: `files/output.csv`
Columnas: `prediccion` y `Probabilidad`

## Contenido
- `inferencia.py`: script de inferencia.
- `preprocesamiento.py`: preprocesamiento de los datos de entrada.
- `requirements.txt`: dependencias necesarias.
- `Dockerfile`: receta de la imagen.
- `preprocessing_training_data.pkl`: objetos de preprocesamiento.
- `pipeline.pkl`: modelo entrenado, guardado con `save_model()`.