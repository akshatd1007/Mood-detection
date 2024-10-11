import streamlit as st
import cv2
import numpy as np

# Title of the app
st.title("Mood Detection App")

# Placeholder for the video feed
video_placeholder = st.empty()

# Function to capture video from webcam
def capture_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Failed to capture video feed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        # Perform mood detection here (placeholder)
        # For example, we just convert to grayscale to simulate processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the video feed
        video_placeholder.image(gray_frame, channels="GRAY", use_column_width=True)

        # Use Streamlit's stop button to break the loop
        if st.button("Stop"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start capturing video on button click
if st.button("Start Video Capture"):
    capture_video()
