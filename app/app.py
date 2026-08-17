import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

'''
Load the trained model once when the app starts (not on every prediction --
st.cache_resource keeps it loaded in memory across user interactions)
'''
@st.cache_resource
def get_model():
    return load_model("models/eye_disease_resnet.keras")

CLASS_NAMES = ["Normal", "Cataract", "Glaucoma", "Retina Disease"]
IMAGE_SIZE = (224, 224)

st.title("Eye Disease Detection")
st.write("Upload a retina image to classify it as normal, cataract, glaucoma, or retina disease.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Image mode:", image.mode, "Image size:", image.size)
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess exactly the same way as training: resize, then ResNet50's
    # own expected preprocessing (not simple 0-1 division, remember why)
    img_resized = image.convert("RGB").resize(IMAGE_SIZE)
    st.write("After conversion:", img_resized.mode, np.array(img_resized).shape)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # model expects a batch, even of size 1
    img_preprocessed = preprocess_input(img_array)

    model = get_model()
    predictions = model.predict(img_preprocessed)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.1f}%")

    '''
    Show all 4 class probabilities, not just the top one -- more transparent,
    and shows when the model is genuinely unsure vs confident
    '''
    st.bar_chart({CLASS_NAMES[i]: float(predictions[i]) for i in range(4)})