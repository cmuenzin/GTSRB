# interface.py
import os
import random
import cv2
import streamlit as st
import numpy as np
from detect_and_crop import detect_and_classify

TEST_IMAGES_DIR = r"C:\Users\casam\Documents\06_Coding\Python\MachineLearning\01_roadSignsProject\data\recognition\Test"

st.title("Verkehrszeichenerkennung GTSRB")
st.write("Lade ein Testbild hoch oder nutze ein zufällig ausgewähltes Bild.")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if st.button("Zufallsbild laden"):
    test_images = [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if test_images:
        random_image_path = random.choice(test_images)
        image = cv2.imread(random_image_path)
        st.success(f"Zufallsbild: {random_image_path}")
    else:
        st.error("Keine Testbilder gefunden!")
        image = None
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
else:
    image = None

if image is not None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Eingegebenes Bild", use_container_width=True)
    temp_path = "temp_test_image.jpg"
    cv2.imwrite(temp_path, image)
    st.write("Verarbeite Bild ...")
    annotated_image = detect_and_classify(temp_path)
    if annotated_image is not None:
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Ergebnis: Annotiertes Bild", use_container_width=True)
    else:
        st.error("Fehler bei der Verarbeitung des Bildes.")
    if os.path.exists(temp_path):
        os.remove(temp_path)
