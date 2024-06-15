import streamlit as st
import tensorflow as tf
import time
import pandas as pd
import random

my_data = {}
my_data['image'] = []
my_data['result'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'images' not in st.session_state:
    st.session_state['images'] = []

model1 = tf.keras.models.load_model("real_vs_ai_model_limitless.h5", compile = False)
model2 = tf.keras.models.load_model("Model_lakshay.h5", compile = False)
val = random.choice([0,1])

st.title("***Analyze Images***")



if 'file' not in st.session_state:
    st.session_state['file'] = True
if 'image' not in st.session_state:
    st.session_state['image'] = None
if st.session_state['file']:
    file = st.file_uploader(label="Upload your photo here ...", type=['png', 'jpeg', 'jpg'])
    if file is not None:
        st.session_state['image'] = file
        st.session_state['file'] = False
        st.rerun()


if st.session_state['image'] is not None:
    col1, col2 = st.columns([3,1])
    col2.header("Options")

    with col2:
        if st.button("Add another Image"):
            st.session_state['file'] = True
            st.session_state['image'] = None
            st.rerun()    

    with col1:
        st.header("Your uploaded image")  
        st.image(st.session_state['image'])
        st.header("Your image is: ")
        image_tensor = tf.io.decode_image(st.session_state['image'].read(), channels=3)
        image_rgb = tf.image.convert_image_dtype(image_tensor, tf.uint8)
        image_save = tf.reshape(image_rgb, [32, 32, 3]).numpy()
        image_test = tf.reshape(image_rgb, [1,32, 32, 3]).numpy()
        if val is 1:
            model = model1
            image_test = image_test/(255/2) - 1
        else:
            model = model2
            image = image_test/255
        with st.spinner("please wait"):
            time.sleep(3)
        if model.predict(image_test) > 0.5:
            st.write("REAL")
        else:
            st.write("AI") 
        st.session_state['images'].append(image_save)
        st.session_state['results'].append("REAL" if model.predict(image_test)>0.5 else "AI")
        my_data['image'] = st.session_state['images']
        my_data['result'] = st.session_state['results'] 

    with col2:
        st.download_button(label = "Download all your image analysis", data = pd.DataFrame(my_data).to_csv(), file_name="image_in_32*32*3.csv", mime = "tsxt/csv")
