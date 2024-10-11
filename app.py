import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

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

# Start video capture
video_capture = cv2.VideoCapture(0)  # Open webcam for video feed

# Streamlit button to start mood detection
if st.button("Start Mood Detection"):
    stframe = st.empty()  # Create an empty Streamlit frame to display the video feed

    while True:
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

        # For local testing, you can stop the loop with 'q' key (comment out if running only in Streamlit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()
