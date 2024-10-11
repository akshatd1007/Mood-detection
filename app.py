import cv2
import streamlit as st

st.title("Webcam Video Feed")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Failed to open camera.")
else:
    # Read frames from the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video feed.")
            break
        
        st.image(frame, channels="BGR")
        # Process the frame here if needed

# Release the camera when done
cap.release()
