import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
#polars
PATH = "cars_train"
#PATH_COLORS = "car_colors"
goku=True

def pred_test(img_path):
   import matplotlib.pyplot as plt
   modelo_marca = tf.keras.models.load_model("modelo_marca.h5")

   img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
   img_arr = tf.keras.preprocessing.image.img_to_array(img).astype("float32")
   plt.imshow(img_arr.astype("uint8"))
   plt.title("Imagen que recibe el modelo")
   plt.show()

 # preprocess
   from tensorflow.keras.applications.efficientnet import preprocess_input
   img_p = preprocess_input(img_arr)
   print("Valores después del preprocess (min, max):", img_p.min(), img_p.max())

   img_p = np.expand_dims(img_p, axis=0)

   pred = modelo_marca.predict(img_p)[0]

   print("Vector de predicciones:")
   print(pred)

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
        
        class_names = train_ds.class_names

        train_ds = train_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        #postprocesamiento: normalización
        train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

        print("Dataset cargado correctamente.")
        return train_ds, val_ds, class_names
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None, None

def verificarDatabase():
 import os
 from PIL import Image

 ruta = "cars_train"

 for carpeta, _, archivos in os.walk(ruta):
    for archivo in archivos:
        path = os.path.join(carpeta, archivo)
        try:
            img = Image.open(path)

            if img.mode != "RGB":
                print(f"Convirtiendo {img.mode} → RGB :", path)
                rgb = img.convert("RGB")
                rgb.save(path)

        except:
            print("❌ No se pudo abrir:", path)




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


def entrenar_modelo(model, train_ds, val_ds,class_names,nombre):
    if train_ds is None:
        print("No se puede entrenar: dataset no cargado.")
        return None

    history = model.fit(train_ds, validation_data=val_ds, epochs=15)
    print("Entrenamiento completado YIPIIII.")

    model.save(f"{nombre}.h5")
    print(f"Modelo {nombre}.h5 guardado.")

    
    with open(f"{nombre}_clases.json", "w") as f:
        json.dump(class_names, f)

    print("Clases guardadas en clases.json")

   
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
            train_ds, val_ds,class_names = cargar_dataset(PATH)
            modelo=crear_modelo(len(class_names))
            entrenar_modelo(modelo, train_ds, val_ds,class_names,"modelo_marca")
            

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
             img = tf.keras.preprocessing.image.img_to_array(img)/255.0

             img = preprocess_input(img) 

             img = np.expand_dims(img, axis=0)
             pred1 = modelo_marca.predict(img)[0]
             top_idx = pred1.argsort()[-3:][::-1]
             for i in top_idx:
              print(f"{marcas[i]}: {pred1[i]:.3f}")
             
             print("Marca: ", marcas[np.argmax(pred1)])
            
            
        case "4":
            goku=False
            print("chao!")
        
        case "5":
         print("DEBUG MODE ACTIVADO")
         #verificarDatabase()
         pred_test("car.ai/cars_test/00153.jpg")






