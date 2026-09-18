import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 🎨 Page Layout & Styling
st.set_page_config(page_title="Smart City ITS", page_icon="🚦", layout="wide")

st.title("🚦 Smart City: Clearway Obstruction Detection")
st.markdown("**Real-time edge computer vision tracking system to identify traffic bottlenecks.**")

st.sidebar.header("⚙️ System Controls")
run_system = st.sidebar.button("Start Edge CV 🚀")
stop_system = st.sidebar.button("Stop System 🛑")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live AI Camera Feed (Tracked)")
    frame_window = st.empty()

with col2:
    st.subheader("🚨 Live Alerts & Analytics")
    status_text = st.empty()
    status_text.info("System Ready. Click 'Start Edge CV' to deploy tracking.")

# 📐 Bus Lane Coordinates (Region of Interest)
BUS_LANE_POLYGON = np.array([
    [150, 600],   
    [350, 300],   
    [700, 300],   
    [900, 600]    
], np.int32)

if run_system:
    status_text.warning("Loading YOLOv9 AI Model with ByteTrack... ⏳")
    
    # 🛑 Yahan se 'try' block shuru hota hai
    try:
        model = YOLO("yolov9c.pt") 
        status_text.success("✅ Tracking Active! Bus Lane monitored.")
        
        cap = cv2.VideoCapture('video.mp4')
        
        while cap.isOpened() and not stop_system:
            success, frame = cap.read()
            if not success:
                break
            
            # AI Tracking
            results = model.track(frame, classes=[2, 5, 7], conf=0.4, persist=True)
            annotated_frame = results[0].plot()
            
            # ROI Draw karna
            cv2.polylines(annotated_frame, [BUS_LANE_POLYGON], isClosed=True, color=(255, 0, 0), thickness=3)
            cv2.putText(annotated_frame, "CLEARWAY ZONE", (360, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # ✅ CORRECTED LINE: use_container_width=True
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        
    # 🛑 Yeh hai woh 'except' block jo miss ho gaya tha
    except Exception as e:
        st.error(f"System Error: {e}")
