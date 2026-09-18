# 🚦 Smart City ITS: Real-Time Edge Computer Vision & Clearway Obstruction Detection

An edge-computing intelligent transportation system (ITS) designed to monitor urban clearways and bus lanes, detect illegal parking bottlenecks in real-time, and generate automated dispatch alerts for municipal authorities.

---

## 🌟 Overview
Urban traffic congestion caused by unauthorized or illegal parking in restricted bus lanes and clearways is a major bottleneck for public transit. This system ingests live camera feeds or CCTV recordings, tracks vehicles using *YOLOv9 & ByteTrack, enforces time-based parking regulations, and visualizes live traffic metrics via a **Streamlit* dashboard.

---

## 🚀 Key Features

* *👀 Real-Time Edge Tracking:* Leverages YOLOv9 object detection paired with ByteTrack (lapx) to assign unique IDs to vehicles and track their trajectories.
* *🛑 Spatial & Stationary Logic:* Calculates pixel-level movement to precisely identify when a vehicle is stalled inside a restricted "Clearway Zone" rather than just passing through.
* *⏰ Time-Based Enforcement Rules:* Dynamic rule engine that accounts for rush hours and off-hours, ensuring alerts are only triggered during active enforcement windows.
* *🔥 Traffic Density Heatmaps:* Visual overlay layer to identify recurring bottleneck zones and high-congestion areas.
* *📊 Live Analytics Dashboard:* Interactive Streamlit layout featuring live metric cards, live vehicle type breakdown charts (Cars, Buses, Trucks), and real-time alert logs.

---

## 🛠️ Tech Stack & Libraries

* *Core Language:* Python 3.10+
* *Computer Vision & AI:* Ultralytics YOLOv9, OpenCV, NumPy
* *Frontend & Dashboard:* Streamlit
* *Data Processing & Export:* Pandas

---

## ⚙️ Project Structure

```text
SmartCity-ITS/
│
├── app.py              # Main Streamlit application & computer vision loop
├── requirements.txt    # Project dependencies for cloud deployment
├── video.mp4           # Sample traffic footage for testing
└── README.md           # Project documentation
