import streamlit as st
import time
import plotly.express as px
import numpy as np
import pandas as pd
from model import get_model, predict_image
from bob import obtener_respuesta

# 1. Configuración de página
st.set_page_config(
    page_title="IA Hackatón SIC 2025",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PARA ANCHO FLEXIBLE (Permite que la flecha funcione) ---
st.markdown("""
    <style>
    /* 1. FONDO DE LA PÁGINA PRINCIPAL */
    .stApp, .stAppHeader {
        background-color: #F5F7F9 !important;
    }

    /* 2. LIMPIEZA TOTAL DEL SIDEBAR */
    /* Buscamos todos los posibles contenedores internos del sidebar */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div, 
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background-color: #F5F7F9 !important;
        border-right: none !important; /* Quita la línea divisoria si quieres */
    }

    /* 3. ANCHO DEL SIDEBAR */
    [data-testid="stSidebar"] {
        width: 450px !important;
    }

    /* 4. ASEGURAR QUE LA FLECHA SE VEA */
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #f0f2f6 !important;
        color: #007bff !important;
    }

    /* 5. RESALTAR EL CHAT */
    /* Como el fondo ahora es gris, las burbujas blancas se verán mucho mejor */
    .stChatMessage {
        background-color: #ffffff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
            /* 1. Efecto de Contenedor de Imagen (Glow Azul) */
[data-testid="stImage"] {
    border: 2px solid #007bff !important;
    border-radius: 15px !important;
    box-shadow: 0 0 20px rgba(0, 123, 255, 0.4) !important;
    padding: 5px !important;
    background-color: rgba(0, 123, 255, 0.05) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

 /* 2. Animación de "Latido" (Opcional, hace que el brillo suba y baje) */
 @keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(0, 123, 255, 0.4); }
    50% { box-shadow: 0 0 25px rgba(0, 123, 255, 0.8); }
    100% { box-shadow: 0 0 10px rgba(0, 123, 255, 0.4); }
 }

 [data-testid="stImage"] {
    animation: pulse 2s infinite;
}

 /* 3. Efecto al pasar el mouse (Hover) */
 [data-testid="stImage"]:hover {
    transform: scale(1.02);
    box-shadow: 0 0 30px rgba(0, 123, 255, 1) !important;
 }
            /* Tarjeta de Identificación del Vehículo */
 .car-card {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px;
    border-left: 10px solid #072357; /* Borde grueso azul */
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
 }

 .car-header {
    color: #072357;
    font-size: 1.2rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 2px solid #f0f2f6;
    margin-bottom: 15px;
    padding-bottom: 5px;
 }

 .car-data {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
 }

 .label {
    font-weight: bold;
    color: #555;
 }

 .value {
    color: #000;
 }
    </style>
    """, unsafe_allow_html=True)

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

if "historial_detecciones" not in st.session_state:
    st.session_state.historial_detecciones = []

# Cargar modelo
@st.cache_resource
def load_all():
    return get_model()
modelo, classes, modelo_color, classes_color = load_all()

# --- CUERPO PRINCIPAL ---
st.title("Car.AI 🚗")
st.caption("Hackatón SIC 2025")

tab1, tab2,tab3 = st.tabs(["🏠 Home", "🧠 Clasificación","📊 Estadisticas"])

with tab1:
    st.title("Car.AI 🚗")
    st.markdown("### La revolución en gestión vial con Inteligencia Artificial")
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.markdown("#### 🔍 Visión")
        st.write("Identificación precisa de marcas, modelos y colores.")
    with col_v2:
        st.markdown("#### ⚖️ Información")
        st.write("Consulta a nuestro bot sobre data de automoviles.")
    with col_v3:
        st.markdown("#### 📈 Datos")
        st.write("Estadísticas en tiempo real para toma de decisiones.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entrada")
        uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"],help="Carga una imagen del vehículo que deseas analizar¨(JPG, PNG)")
        if uploaded:
            st.image(uploaded, use_container_width=True)
    with col2:
        st.subheader("Resultado")
        if uploaded and st.button("🚀 Analizar", use_container_width=True,help="Determinar marca y color del vehiculo cargado"):
            with st.spinner("Analizando..."):
                preds_m, preds_c = predict_image(modelo, modelo_color, uploaded)
                # Obtener etiquetas
                label_m = classes[str(preds_m.argmax())]
                label_c = classes_color[str(preds_c.argmax())]
                
                probabilidad = np.max(preds_m) * 100

                st.session_state.historial_detecciones.append({
            "Marca": label_m,
            "Color": label_c
        })
    
                st.success(f"Veo un **{label_m}** de color **{label_c}**")
                st.markdown(f"""
<div class="car-card">
    <div class="car-header">📋 Reporte de Detección Car.AI</div>
    <div class="car-data">
        <span class="label">Modelo/Marca:</span>
        <span class="value">{label_m}</span>
    </div>
    <div class="car-data">
        <span class="label">Color Detectado:</span>
        <span class="value">{label_c}</span>
    </div>
    <div class="car-data">
        <span class="label">Confianza IA:</span>
        <span class="value">{probabilidad:.1f}%</span>
    </div>
    <div class="car-data">
        <span class="label">ID de Sesión:</span>
        <span class="value">#SIC-2025-{np.random.randint(1000, 9999)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

with tab3:
 st.header("📊 Estadísticas de la Sesión")
 if len(st.session_state.historial_detecciones) > 0:
        df = pd.DataFrame(st.session_state.historial_detecciones)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Marcas detectadas")
            counts_m = df["Marca"].value_counts().reset_index()
            # Creamos el gráfico con color específico (Azul Panamá)
            fig_m = px.bar(counts_m, x='Marca', y='count', 
                           color_discrete_sequence=['#072357']) 
            fig_m.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_m, use_container_width=True)
            
        with col_b:
            st.subheader("Colores detectados")
            counts_c = df["Color"].value_counts().reset_index()
            # Creamos el gráfico con otro color (Rojo Panamá)
            fig_c = px.bar(counts_c, x='Color', y='count', 
                           color_discrete_sequence=['#da121a'])
            fig_c.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_c, use_container_width=True)
            
        st.write("### Registro detallado")
        st.dataframe(df, use_container_width=True)
 else:
        st.info("Aún no hay datos. Ve a la pestaña 'Clasificación' y analiza una imagen.")
# --- EL CHAT EN EL SIDEBAR ---
with st.sidebar:
    st.header("💬 Bob")
    
    # Usar el contenedor con scroll
    chat_container = st.container(height=550)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu duda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                 response = obtener_respuesta(prompt)
                 def stream_data():
                  for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.05) # Ajusta este valor para más/menos velocidad

            # 3. Usamos st.write_stream para el efecto visual
            placeholder_response = st.write_stream(stream_data)   
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})