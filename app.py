import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import joblib

# Load model and label encoder
model = tf.keras.models.load_model("twohand_model.h5")
encoder = joblib.load("label_encoder.pkl")

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# TTS engine
engine = pyttsx3.init()
last_prediction = None

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    landmark_list = []

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(landmark_list) == 126:  # 21 landmarks * 3 coords * 2 hands
            input_data = np.array(landmark_list).reshape(1, -1)
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            class_name = encoder.inverse_transform([predicted_class])[0]

            cv2.putText(frame, f"Sign: {class_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if last_prediction != class_name:
                engine.say(class_name)
                engine.runAndWait()
                last_prediction = class_name

    else:
        cv2.putText(frame, "Show both hands clearly", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("BSL Two-Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
