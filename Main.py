import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
#polars
PATH = "cars_train"
#PATH_COLORS = "car_colors"
goku=True

def cargar_dataset(PATH):
#Aqui dentro se encuentran todas las fotos que se van a usar para entrenar el modelo
    print(f"Cargando dataset: "+PATH+" ...")
    try:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            PATH,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(224, 224),
            batch_size=32,
            label_mode="categorical"
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            PATH,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(224, 224),
            batch_size=32,
            label_mode="categorical"
        )
        print("Dataset cargado correctamente.")
        return train_ds, val_ds
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None, None

#num_clases es la cantidad de marcas de autos que se van a clasificar
def crear_modelo(num_clases):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))

    # congelar la base
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    out = Dense(num_clases, activation="softmax")(x)

    model = Model(base.input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("Modelo creado correctamente.")
    return model


def entrenar_modelo(model, train_ds, val_ds,nombre):
    if train_ds is None:
        print("No se puede entrenar: dataset no cargado.")
        return None

    history = model.fit(train_ds, validation_data=val_ds, epochs=15)
    print("Entrenamiento completado YIPIIII.")

    
    with open(f"{nombre}_clases.json", "w") as f:
        json.dump(train_ds.class_names, f)

    print("Clases guardadas en clases.json")

    model.save(f"{nombre}.h5")
    print(f"Modelo {nombre}.h5 guardado.")
    return history








while goku:
    print(*"="*50)
    print("*ALERTA: ESTE ALOGIRTMO USA BASTANTE RECURSOS DE TU MAQUINCA! ")
    print(*"="*50)
    print ("1. Entrenar modelo de marca (*)")
    print("2. Entrenar modelo de color (*)")
    print("3. Realizar predicción")
    print("4. Salir")
    opcion = input("Selecciona una opción: ")
    match opcion:
        case "1":
            train_ds, val_ds = cargar_dataset(PATH)
            modelo=crear_modelo(20)
            entrenar_modelo(modelo, train_ds, val_ds,"modelo_marca")
            

        case "2":
            print("Funcionalidad de color no implementada aún.")
            #train_ds, val_ds = cargar_dataset(PATH_COLORS)
            #modelo=crear_modelo(10)
            #entrenar_modelo(modelo, train_ds, val_ds,"modelo_color")
        case "3":
             try:
              modelo_marca = tf.keras.models.load_model("modelo_marca.h5")
              #modelo_color = tf.keras.models.load_model("modelo_color.h5")
             except Exception as e:
                print(f"Error cargando el modelo: {e}")
                continue
             try:
              with open("modelo_marca_clases.json", "r") as f:
               marcas = json.load(f)
               #with open("modelo_color_clases.json", "r") as f:
                #color_class_names = json.load(f)
             except:
              print("No se encontró clases.json, no se puede interpretar las predicciones.")
              continue
             imagen=input("Elije la imagen para predecir")
             img = tf.keras.preprocessing.image.load_img(imagen, target_size=(224,224))
             img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
             img = np.expand_dims(img, axis=0)
             pred1 = modelo_marca.predict(img)
             
             print("Marca: ", marcas[np.argmax(pred1)])
            
            
        case "4":
            goku=False
            print("chao!")
            


