# 🚗 AI-Based Autonomous Navigation System (Software)

👩‍💻 Developed by: Dhananjay Bhaskar

---

## 📌 Project Overview

This project is a software-based simulation of an AI-powered autonomous navigation system. It detects objects, identifies lanes, and makes movement decisions similar to a self-driving vehicle.

Due to budget constraints, this implementation focuses only on the software part using a webcam.

---

## 🎯 Features

- ✅ Real-time object detection (YOLOv3)
- ✅ Lane detection using OpenCV
- ✅ Decision making (STOP / SLOW / CLEAR)
- ✅ Path planning (LEFT / RIGHT / STRAIGHT)
- ✅ Distance estimation (visual-based)
- ✅ Live camera processing

---

## 🧠 How It Works

1. Camera captures live video
2. YOLO detects objects (person, car, etc.)
3. Lane detection identifies road direction
4. AI makes decisions:
   - Person → STOP
   - Vehicle → SLOW
   - Clear path → GO STRAIGHT
5. System displays movement instructions

---

## 🛠️ Technologies Used

- Python
- OpenCV
- YOLOv3
- NumPy

---

## ▶️ How to Run

1. Install dependencies:

pip install opencv-python numpy


2. Download YOLO weights:
https://pjreddie.com/media/files/yolov3.weights

3. Place files in same folder:
- object_detection.py
- yolov3.cfg
- coco.names
- yolov3.weights

4. Run:

python object_detection.py



---

## 🚀 Future Scope

- Raspberry Pi integration
- Ultrasonic sensor for accurate distance
- Motor control for real robot movement
- Fully autonomous vehicle prototype

---

## 📄 Conclusion

This project demonstrates a basic autonomous navigation system using AI and computer vision. It can be extended to real-world applications with hardware integration.

---
