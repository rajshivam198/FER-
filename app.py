import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json

# Load Model
json_file = open("faceemotiondetection.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("faceemotiondetection.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    features = np.array(image).reshape(1, 48, 48, 1)
    return features / 255.0

st.title("Real-Time Facial Emotion Recognition")
run = st.checkbox('Start Webcam')

if run:
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            emotion = labels[pred.argmax()]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_window.image(frame, channels="BGR")

    cap.release()
