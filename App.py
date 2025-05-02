import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.markdown("<style>body { background-color: #E0D585; margin: 0; padding: 0; }</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Reconocimiento de Dígitos escritos a mano</h1>", unsafe_allow_html=True)

# Mostrar la imagen debajo del título
st.image("DIBUJITO.png", use_container_width=True)

st.markdown("<h3 style='text-align: center;'>Dibuja el digito en el panel y presiona 'Predecir'</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>En esta aplicación se evalua la capacidad de un RNA de reconocer dígitos escritos a mano. Basado en desarrollo de Vinay Uniyal.</p>", unsafe_allow_html=True)

drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas",
    )

with col2:
    if st.button('Predecir'):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
            input_image.save('prediction/img.png')
            img = Image.open("prediction/img.png")
            res = predictDigit(img)
            st.markdown(f"<h2 style='text-align: center;'>El Dígito es: {res}</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center;'>Por favor dibuja en el canvas el dígito.</h2>", unsafe_allow_html=True)
