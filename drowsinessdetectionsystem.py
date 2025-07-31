import cv2
import numpy as np
from keras.models import load_model

model = load_model("drowsiness_detection_model.h5")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_drowsiness(eye_img, threshold=0.25):
    try:
        gray = cv2.resize(eye_img, (48,48))
        gray = np.expand_dims(gray, axis=(0,-1))
        gray = gray.astype("float32")/255.0
        prediction = model.predict(gray, verbose=0)
        print("prediction (closed,open):", prediction[0])
        confidence_open = prediction[0][1]
        if confidence_open <= threshold:
            return "Safe to Drive", confidence_open
        else: 
            return "Drowsiness detected, find the closest rest area", 1-confidence_open
    except Exception as e:
        return "Error", 0.0
    
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam")
    exit()
while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error: Failed to capture frame")
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.1, 4)
    label = "No Eyes Detected"
    confidence = 0.0
    color = (255,255,0)
    for (x,y,w,h) in eyes:
        eye_region = gray_frame[y: y+h, x: x+w]
        label, confidence = detect_drowsiness(eye_region)
        color = (0, 255, 0) if "Safe" in label else (0, 0, 255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        break
    cv2.putText(frame, f"{label} ({confidence:.2f})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Real Time Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

