import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

st.set_page_config(page_title='🎨 Dibuja un Número', layout='wide')

# Estilos personalizados para una experiencia más colorida y divertida
st.markdown("""
    <style>
    body {
        background-color: #FFF3E0;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    .stApp {
        background-color: #FFF3E0;
    }

    h1, h2, h3 {
        color: #D84315;
        text-align: center;
    }

    .stButton>button {
        background-color: #FF7043;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.5em 1em;
    }

    .stButton>button:hover {
        background-color: #FF5722;
    }

    .stSlider > div {
        color: #D84315;
    }

    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stSidebar {display: none;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎨 ¡Dibuja un número y lo adivinamos!</h1>", unsafe_allow_html=True)
st.markdown("<h3>🖌️ Usa el panel de dibujo para escribir un número del 0 al 9.<br>Luego haz clic en 'Predecir' para ver si la IA acierta 🎯</h3>", unsafe_allow_html=True)

stroke_width = st.slider("✏️ Ancho del lápiz", 1, 30, 15)
stroke_color = "#FFFFFF"
bg_color = "#000000"

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='text-align:center;'>🖼️ Panel de dibujo</h3>", unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas",
    )

with col2:
    st.markdown("<h3 style='text-align:center;'>🤖 Resultado</h3>", unsafe_allow_html=True)
    if st.button("✨ Predecir"):
        if canvas_result.image_data is not None:
            input_array = np.array(canvas_result.image_data)
            img = Image.fromarray(input_array.astype('uint8'), 'RGBA')
            res = predictDigit(img)
            st.markdown(f"<h2 style='text-align:center;'>🧠 ¡Creo que es un <span style='color:#D84315;'>{res}</span>!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align:center;'>🚨 ¡Primero dibuja algo!</h2>", unsafe_allow_html=True)
