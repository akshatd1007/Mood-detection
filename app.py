import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading

# Load pre-trained emotion detection model
model = load_model('emotion_model.hdf5', compile=False)  # Update this to your model's path
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Emotion labels based on your model's training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to process the video feed and predict mood
def detect_mood(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Resize to 64x64 pixels as required by the model
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to range [0,1]
    reshaped_frame = np.reshape(normalized_frame, [1, 64, 64, 1])  # Reshape to (1, 64, 64, 1) for prediction
    prediction = model.predict(reshaped_frame)  # Get prediction from the model
    return emotion_labels[np.argmax(prediction)]  # Return the label with the highest probability

# Streamlit application layout
st.title("Real-Time Mood Detection")
st.write("Using facial expression analysis to detect mood in real-time.")

# Video capture setup
video_capture = cv2.VideoCapture(1)  # Open webcam for video feed
stframe = st.empty()  # Create an empty Streamlit frame to display the video feed

def capture_video():
    while st.session_state.running:
        ret, frame = video_capture.read()  # Capture frame-by-frame
        if not ret:
            st.write("Failed to capture video feed.")
            break

        # Detect mood
        mood = detect_mood(frame)
        
        # Display the detected mood on the frame
        cv2.putText(frame, mood, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame in Streamlit
        stframe.image(frame, channels='BGR')

# Streamlit button to start mood detection
if st.button("Start Mood Detection"):
    st.session_state.running = True
    video_thread = threading.Thread(target=capture_video)
    video_thread.start()

# Streamlit button to stop mood detection
if st.button("Stop Mood Detection"):
    st.session_state.running = False
    video_capture.release()

# On app close, release the video capture
if 'running' in st.session_state and not st.session_state.running:
    video_capture.release()
