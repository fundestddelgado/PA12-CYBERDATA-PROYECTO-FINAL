import tensorflow as tf
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

@tf.keras.utils.register_keras_serializable()
def get_model():
    # Carga de modelos
    modelo = tf.keras.models.load_model("Hackaton SIC 2025/modelo_marca_tipo.h5")
    modelo_color = tf.keras.models.load_model("Hackaton SIC 2025/modelo_color.h5")

    # Carga de clases para marca/tipo
    try:
        with open("Hackaton SIC 2025/modelo_marca_tipo_clases.json") as f:
            classes = json.load(f)
    except:
        classes = None

    # Carga de clases para color
    try:
        with open("Hackaton SIC 2025/modelo_color_clases.json") as f:
            classes_color = json.load(f)
    except:
        classes_color = None

    return modelo, classes, modelo_color, classes_color


def predict_image(model, modelo_color, img_file):
    # Preprocesamiento de imagen
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predicciones
    preds_marca = model.predict(img)
    preds_color = modelo_color.predict(img)
    
    return preds_marca, preds_color