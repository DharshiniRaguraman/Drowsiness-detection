# 💤Drowsiness-detection
A real-time drowsiness detection system using OpenCV and deep learning. Detects eye closure through webcam and alerts the user with sound if drowsiness is detected. Ideal for driver safety and monitoring fatigue in real-time applications.

## 📌 Overview

Drowsy driving is one of the leading causes of road accidents. This project implements a computer vision-based solution to detect signs of drowsiness—especially eye closure duration—using webcam input. If drowsiness is detected, an alert is triggered to wake the user.


## 🎯 Features

Real-time eye tracking using webcam

EAR (Eye Aspect Ratio) based drowsiness detection

Audio alert system on detecting prolonged eye closure

Lightweight and fast enough to run on most systems

Modular and easy-to-extend codebase


## 🧠 Techniques Used

Computer Vision with OpenCV

Facial Landmark Detection using dlib

Eye Aspect Ratio (EAR) computation

Machine Learning : CNN-based eye state classification (open/closed)

Multithreading for smoother video processing



## 🛠 Requirements

Install the required libraries using pip:

pip install opencv-python dlib imutils scipy playsound

Additional setup:

shape_predictor_68_face_landmarks.dat file (download from dlib’s official source)



## 🚀 How It Works

1. Captures real-time video through webcam.


2. Detects face and facial landmarks using dlib.


3. Calculates EAR (Eye Aspect Ratio).


4. If EAR falls below a threshold for a certain number of consecutive frames, it detects drowsiness.


5. Triggers an alert sound to wake the user.

## 📁 Project Structure

drowsiness-detection/

│

├── drowsiness_detection_system.py

├── drowsiness_detection_model.py

├── drowsiness_detection_model.h5

├── requirements.txt

└── README.md


## 📊 Customization

You can tune the following parameters in the code:

EAR_THRESHOLD: Threshold below which eye is considered closed

CONSEC_FRAMES: Number of consecutive frames indicating drowsiness


You can also extend the project to include:

Yawning detection

Head pose estimation

CNN-based eye state classification


## 🧩 Future Improvements

Add mobile/IoT support (e.g., Raspberry Pi)

Deploy using Flask or Streamlit for UI

Integrate with car systems for real-world alerts

### 📌 Author

Made with ❤️ by Dharshini

Feel free to fork, improve, or contribute!
