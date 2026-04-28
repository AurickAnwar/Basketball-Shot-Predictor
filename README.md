🏀 Basketball Shot Analyzer
A computer vision + machine learning project that analyzes basketball shots from video and predicts whether the shot will go in.
🚀 Features
Detects basketball using YOLOv11
Tracks ball position frame-by-frame
Calculates motion features:
Position (x, y)
Velocity (vx, vy)
Distance from rim
Speed
Logs data into a CSV file
Uses a PyTorch model to predict shot success
Displays predictions in real-time on video

🧠 How It Works
Object Detection
Uses YOLO to detect the basketball in each frame
Tracking + Physics
Computes:
Δx, Δy
Velocity
Distance to rim
Detects when ball crosses rim plane
Data Collection
Stores frame-by-frame data into Final_Shots.csv
Machine Learning
Trains a neural network on shot data
Outputs probability of shot being made

📁 Project Structure
Basketball-Shot-Analyzer/
│── main.py                # YOLO + OpenCV tracking
│── model.py               # PyTorch model
│── Final_Shots.csv        # Dataset
│── videos/                # Input videos
│── README.md

🛠️ Tech Stack
Python
OpenCV
Ultralytics YOLOv11
PyTorch
NumPy / Pandas

⚙️ Installation
pip install ultralytics opencv-python torch pandas

▶️ Usage
Place your video file:
vid.mp4
Run detection:
python main.py
Train model:
python model.py

📊 Example Features (CSV)
frame
ball_x
ball_y
vx
vy
speed
label
12
320
210
5
-3
5.8
1

label = 1 → shot made
label = 0 → shot missed

🎯 Future Improvements
Improve ball detection accuracy
Use multiple videos for better training
Add trajectory prediction visualization
Real-time shot feedback system
Train more advanced models (LSTM / sequence-based)

💡 Inspiration
This project combines computer vision + physics + machine learning to simulate how real analytics systems track and predict basketball shots.

🧑‍💻 Author
Built by Aurick Anwar

## 🚀 Demo

![Preview](https://img.youtube.com/vi/B-A5uHzQIgI/0.jpg)

▶️ Full Demo: https://www.youtube.com/watch?v=B-A5uHzQIgI


