import streamlit as st
import cv2
import numpy as np

# Function to process images (replace this with your actual processing logic)
def process_image(image):
    # Example processing: convert to grayscale (or your mood detection logic)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Streamlit UI
st.title("Mood Detection App")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read the image

    if image is not None:
        # Display the original image
        st.image(image, caption='Original Image', use_column_width=True)

        # Process the image
        processed_image = process_image(image)

        # Display the processed image
        st.image(processed_image, caption='Processed Image', use_column_width=True)
    else:
        st.error("Could not read the image.")
