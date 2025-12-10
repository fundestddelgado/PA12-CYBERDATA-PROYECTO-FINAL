import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
img_analisis=""
main= tk.Tk()
main.resizable(False, False)
main.title("Proyecto 2 - Car.ai")
main.geometry("1000x550")
main.configure(bg="#F0F2F5")
#icono = tk.PhotoImage(file="ProjectoSamPython/logoprueba2.PNG") 
#main.iconphoto(True, icono)


# Función que se ejecuta al presionar Enter

def paginaHome():
    # Frame principal con tamaño fijo
    homeFrame = tk.Frame(main_frame)
    homeFrame.pack_propagate(False)
    homeFrame.pack(fill="both", expand=True,padx=0, pady=0)   
    

    # Imagen de fondo
    fondo = tk.PhotoImage(file="Swerving Car Meme.png")
    #placeholder temporal
    label_fondo = tk.Label(homeFrame, image=fondo)
    label_fondo.image = fondo  # Mantener referencia
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)  


def paginaAnalisis():
    analisisFrame = tk.Frame(main_frame)
    analisisFrame.pack_propagate(False)
    analisisFrame.pack(fill="both", expand=True,padx=0, pady=0) 
    def crear_tooltip(widget, texto):
     tooltip = tk.Toplevel(widget)
     tooltip.withdraw()  # Oculto por defecto
     tooltip.overrideredirect(True)  # Sin bordes ni barra de título
     label = tk.Label(tooltip, text=texto, background="cyan", relief="solid", borderwidth=1)
     label.pack()

     def mostrar_tooltip(event):
        tooltip.deiconify()
        tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

     def ocultar_tooltip(event):
        tooltip.withdraw()

     widget.bind("<Enter>", mostrar_tooltip)
     widget.bind("<Leave>", ocultar_tooltip)

    # Primero creamos la foto con Pillow
    fondoAnalisis = Image.open("AnalisisBg.png")
    fondoAnalisis = fondoAnalisis.resize((1000, 700))
    fondoAnalisis = ImageTk.PhotoImage(fondoAnalisis)  
    label_fondo = tk.Label(analisisFrame, image=fondoAnalisis)
    label_fondo.image = fondoAnalisis  # Mantener referencia
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)
    #Yea ahora con el fondo nuevo, los labeles se ven feos, 
    #mejor incorporarlos dentro del mismo fondo
    titulo = tk.Label(analisisFrame, text="Inserte una imagen para comenzar ", font=("Arial", 20), bg="Black",fg="white")
    titulo.pack(pady=10)

  

    #def listarEmpresas():
        #print(data1.columns)
        #empresas=data1["empresa"].unique().tolist()
        #return empresas


   

    mensaje_error = tk.Label(analisisFrame, text="", font=("Arial", 12), fg="red", bg="Black")
    mensaje_error.pack(pady=5)


    # Función para generar boxplot
    
   

    # Función para obtener imagen del Entry
    def guardar_imagen():
       ruta=filedialog.askopenfilename(title="Seleccionar imagen", 
        filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.webp;*")])
       if ruta:
            print("Imagen seleccionada:", ruta)
            img_analisis=ruta
            return img_analisis
           

       
    def analizar_imagen():
       if img_analisis =="":
          mensaje_error.config(text="Debes insertar una imagen", fg="red")
       else :
          print("Cargando modelos....")
          try:
           modelo_marca = tf.keras.models.load_model("modelo_marca.h5")
          except Exception as e:
           mensaje_error.config(text=f"Error cargando el modelo: {e}", fg="red")
           return
          print("Modelos cargados correctamente.")
          print("Cargando clases....")
          try:
              with open("modelo_marca_clases.json", "r") as f:
                marcas = json.load(f)
          except:
              mensaje_error.config(text="No se encontró clases.json, no se puede interpretar las predicciones.", fg="red")
              return
          print("Clases cargadas correctamente.")
          img = tf.keras.preprocessing.image.load_img(img_analisis, target_size=(224,224))
          img = tf.keras.preprocessing.image.img_to_array(img)

          img = preprocess_input(img) 

          img = np.expand_dims(img, axis=0)
          pred1 = modelo_marca.predict(img)[0]
          top_idx = pred1.argsort()[-3:][::-1]
          for i in top_idx:
            print(f"{marcas[i]}: {pred1[i]:.3f}")
             
            print("Marca: ", marcas[np.argmax(pred1)])




           
  
    
    


    

    boton_guardar = tk.Button(analisisFrame, text="Insertar imagen", command=guardar_imagen,width=25,height=3)
    boton_guardar.grid(row=0, column=0, padx=10, pady=10)

    crear_tooltip(boton_guardar, "Inserte una imagen para analizar")

    analizar_general= tk.Button(analisisFrame, text="¡Analizar Imagen!", command=analizar_imagen,width=25,height=3)
    analizar_general.grid(row=0, column=1, padx=10, pady=10)

    crear_tooltip(analizar_general, "Analizar la imagen insertada con los modelos cargados")

    
    

def limpiarFrame():
    for widget in main_frame.winfo_children():
        widget.destroy()

def limpiarIndicadores():
    homeIndicador.config(bg="Black")
    analisisIndicador.config(bg="Black")
    

def indicador(lbl,pagina):
    limpiarIndicadores()
    lbl.config(bg="RoyalBlue")
    limpiarFrame()
    pagina()



opcionFrame= tk.Frame(main,bg="black")

#Botton home
homeBotton=tk.Button(opcionFrame,text="Home",font=("Arial", 15),fg="white",bd=0,bg="black"
                     ,command=lambda:indicador(homeIndicador,paginaHome))
homeBotton.place(x=20,y=50)
homeIndicador=tk.Label(opcionFrame,text="",bg="Black")
homeIndicador.place(x=3,y=50,width=5,height=40)
#Botton Analisis
analisisBotton=tk.Button(opcionFrame,text="Análisis",font=("Arial", 15),fg="white",bd=0,bg="black",
                         command=lambda:indicador(analisisIndicador,paginaAnalisis))
analisisBotton.place(x=20,y=100)
analisisIndicador=tk.Label(opcionFrame,text="",bg="Black")
analisisIndicador.place(x=3,y=100,width=5,height=40)
#Botton salir
salirBotton=tk.Button(opcionFrame,text="Salir",font=("Arial", 15),fg="white",bd=0,bg="black",command=main.quit)
salirBotton.place(x=20,y=450)
#Atributos del frame de opciones
opcionFrame.pack(side="left")
#logo = tk.PhotoImage(file="ProjectoSamPython/cybersentinelLogo.png")
#logoChico = logo.subsample(4, 5)  # Ajusta los valores según el tamaño deseado
#logo_label = tk.Label(opcionFrame, image=logoChico, bg="black")
#logo_label.place(x=1, y=5)
opcionFrame.pack_propagate(False)
opcionFrame.config(width=145,height=700)
#Atributos del frame principal
main_frame= tk.Frame(main,highlightbackground="black", highlightthickness=1)
main_frame.pack(side="left")
main_frame.pack_propagate(False)
main_frame.config(width=1000,height=700)
main.configure(padx=0, pady=0) 


main.mainloop()