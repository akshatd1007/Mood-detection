import streamlit as st
import cv2
import numpy as np

st.title("Real-Time Image Capture")

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    st.error("Failed to capture video feed")
else:
    # Create a placeholder for the video feed
    frame_placeholder = st.empty()

    # Capture video frame by frame
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            st.error("Failed to grab frame")
            break
        
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Allow user to capture an image
        if st.button("Capture Image"):
            st.image(frame_rgb, caption="Captured Image", channels="RGB")
            break

# Release the webcam when done
video_capture.release()
cv2.destroyAllWindows()

