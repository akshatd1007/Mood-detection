import streamlit as st
import cv2
import numpy as np

# Function to process images (currently does minimal processing)
def process_image(image):
    # Just return the image without processing for now
    return image

# Streamlit UI
st.title("Mood Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_COLOR to read the image

    if image is not None:
        # Display the original image
        st.image(image, caption='Original Image', use_column_width=True)

        # Process the image
        processed_image = process_image(image)

        # Display the processed image (which is the same as original for now)
        st.image(processed_image, caption='Processed Image', use_column_width=True)
    else:
        st.error("Could not read the image.")
