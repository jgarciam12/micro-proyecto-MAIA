import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Clasificador de Radiografías",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Clasificación de Radiografías de Columna")
st.write("Carga una radiografía y obtén la predicción del modelo.")

uploaded_file = st.file_uploader(
    "Sube una imagen de radiografía",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("🔍 Predecir"):
        with st.spinner("Analizando imagen..."):

            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Predicción completada")

                st.metric("Diagnóstico", result["class_name"])
                st.metric("Confianza", f'{result["confidence"]*100:.2f}%')

            else:
                st.error("❌ Error al comunicarse con la API")
                st.text(response.text)

# 1. Ejecutar: uvicorn deployment.fastap_app.main:app --reload
# 2. Ejecutar: streamlit run deployment/streamlit_app/app_front.py
