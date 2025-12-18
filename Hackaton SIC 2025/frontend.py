import streamlit as st
from model import get_model, predict_image
from bot import obtener_respuesta

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
    </style>
    """, unsafe_allow_html=True)

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cargar modelo
@st.cache_resource
def load_all():
    return get_model()
modelo, classes, modelo_color, classes_color = load_all()

# --- CUERPO PRINCIPAL ---
st.title("Car.AI 🚗")
st.caption("Hackatón SIC 2025")

tab1, tab2 = st.tabs(["🏠 Home", "🧠 Clasificación"])

with tab1:
    st.markdown("## Bienvenido a CAR.AI")
    st.write("Para clasificar una imagen de un vehículo, ve a la pestaña 'Clasificación'.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entrada")
        uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png"])
        if uploaded:
            st.image(uploaded, use_container_width=True)
    with col2:
        st.subheader("Resultado")
        if uploaded and st.button("🚀 Analizar", use_container_width=True):
            with st.spinner("Analizando..."):
                preds_m, preds_c = predict_image(modelo, modelo_color, uploaded)
                # Obtener etiquetas
                label_m = classes[str(preds_m.argmax())]
                label_c = classes_color[str(preds_c.argmax())]
    
                st.success(f"Veo un **{label_m}** de color **{label_c}**")

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
                 st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})