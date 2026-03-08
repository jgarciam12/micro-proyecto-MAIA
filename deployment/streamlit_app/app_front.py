import streamlit as st
import requests
from PIL import Image
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Clasificador de Radiografías",
    page_icon="🦴",
    layout="wide"
)

st.markdown("""
<style>

/* Oculta la barra blanca superior */
header[data-testid="stHeader"]{
    display:none;
}

/* Reduce espacio superior */
.block-container{
    padding-top:0rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"<h1 style='text-align:center;'>🦴 Clasificador de Radiografías de Columna</h1>",
unsafe_allow_html=True
)

# ---------- CSS ----------
st.markdown("""
<style>

.stApp{
    background-color:#0e1117;
    color:white;
}

.card{
    background-color:#1c1f26;
    padding:25px;
    border-radius:15px;
}

.title{
    font-size:40px;
    font-weight:bold;
}

/* estilo para la radiografía */
img{
    border-radius:10px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.6);
}

/* Botones oscuros */
div.stButton > button {
    background-color: #262730;
    color: white;
    border-radius: 8px;
    border: 1px solid #3a3a3a;
    padding: 8px 18px;
}

/* Hover del botón */
div.stButton > button:hover {
    background-color: #3a3f4b;
    color: white;
    border: 1px solid #5a5a5a;
}

</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:
    st.markdown(
        """
        **Sobre esta aplicación**

        Esta aplicación utiliza un modelo de inteligencia artificial para **clasificar
        radiografías de columna**.  

        El sistema puede identificar tres posibles diagnósticos:

        - **Normal**
        - **Scoliosis**
        - **Spondylolisthesis**
        """
    )


with col2:
    st.markdown(
        """      
        Además, el modelo muestra el **nivel de confianza de la predicción**, que indica
        qué tan seguro está el modelo sobre el diagnóstico realizado.
        
        **Cómo usar la aplicación**

        1️⃣ Sube una radiografía  
        2️⃣ Presiona **Analizar imagen**  
        3️⃣ El modelo mostrará el diagnóstico y su probabilidad
        """
    )

col1, col2 = st.columns([1,1])

uploaded_file = None

# ---------- COLUMNA IMAGEN ----------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Sube una radiografía",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        c1, c2, c3 = st.columns([1, 6, 1])

        with c2:
            st.image(
                image,
                caption="Radiografía cargada",
                width=475
            )

st.markdown('</div>', unsafe_allow_html=True)

# ---------- RESULTADOS ----------
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)


    st.subheader("Resultado del modelo")

    if uploaded_file:

        b1, b2 = st.columns(2)

        with b1:
            analizar = st.button("🔍 Analizar imagen")

        with b2:
            nueva = st.button("🔄 Nueva imagen")

        if nueva:
            st.rerun()

        if analizar:

            with st.spinner("Analizando radiografía..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:

                    result = response.json()

                    st.success("Predicción completada")

                    metric1, metric2 = st.columns(2)
                    metric1.metric(
                        "Diagnóstico",
                        result["class_name"]
                    )
                    metric2.metric(
                        "Confianza",
                        f'{result["confidence"]*100:.2f}%'
                    )

                    st.write("### Probabilidades")

                    probs = pd.DataFrame(
                        list(result["probabilities"].items()),
                        columns=["Clase","Probabilidad"]
                    )

                    fig = px.bar(
                        probs,
                        x="Clase",
                        y="Probabilidad",
                        color="Clase",
                        template="plotly_dark"
                    )

                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        yaxis_title="Probabilidad",
                        xaxis_title="Clase"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error("Error al comunicarse con la API")

    else:
        st.info("Sube una imagen para obtener una predicción.")

    st.markdown('</div>', unsafe_allow_html=True)

# 1. Ejecutar: uvicorn deployment.fastap_app.main:app --reload
# 2. Ejecutar: streamlit run deployment/streamlit_app/app_front.py
