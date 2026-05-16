import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

# Title
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI scan image to check whether a tumor is present.")

# Load trained model
try:
    model = load_model("best_brain_tumor_model.keras")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display smaller image
    st.image(
        image,
        caption="Uploaded MRI Scan",
        width=300
    )

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(img_array, verbose=0)[0]

        predicted_index = np.argmax(prediction)

        class_labels = ["No Tumor", "Tumor Detected"]
        result = class_labels[predicted_index]

        confidence = prediction[predicted_index] * 100

        if predicted_index == 1:
            st.error(
                f"🧠 Result: {result}\n\nConfidence: {confidence:.2f}%"
            )
        else:
            st.success(
                f"✅ Result: {result}\n\nConfidence: {confidence:.2f}%"
            )