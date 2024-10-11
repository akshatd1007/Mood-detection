import streamlit as st
import cv2
import numpy as np

# Function to process images
def process_image(image):
    try:
        # Convert the image to grayscale (or any other processing)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit UI
st.title("Mood Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        # Process the image
        processed_image = process_image(image)

        # Display the processed image
        st.image(processed_image, caption='Processed Image', use_column_width=True)
        
        # Optionally, display the original image
        st.image(image, caption='Original Image', use_column_width=True)
    else:
        st.error("Could not read the image.")

# Any additional logic or processing can go here
