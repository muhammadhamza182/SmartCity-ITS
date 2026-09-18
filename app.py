import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# 🎨 Premium Page Layout
st.set_page_config(page_title="Clearway AI Ops", page_icon="🚦", layout="wide")

# 🪧 Header Section
st.title("🚦 Smart City: Edge CV Traffic Management")
st.markdown("**Real-time obstruction tracking and automated dispatch alerts for intelligent transportation systems.**")
st.divider()

# ⚙️ Sidebar: Dynamic Client Inputs
st.sidebar.header("⚙️ Data Source")
source_type = st.sidebar.radio("Select Feed Type:", ["Upload Video", "Live Camera (RTSP)"])

video_path = None
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload CCTV Footage (MP4)", type=['mp4'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif source_type == "Live Camera (RTSP)":
    video_path = st.sidebar.text_input("Enter RTSP Stream Link:", "rtsp://username:password@ip_address/stream")

st.sidebar.markdown("---")
run_system = st.sidebar.button("Deploy AI Tracking 🚀", use_container_width=True)
stop_system = st.sidebar.button("Stop System 🛑", use_container_width=True)

# 📊 Dashboard Metrics UI
m1, m2, m3 = st.columns(3)
with m1:
    total_vehicles_metric = st.empty()
    total_vehicles_metric.metric("Total Vehicles Tracked", "0")
with m2:
    lane_occupants_metric = st.empty()
    lane_occupants_metric.metric("Vehicles in Clearway", "0")
with m3:
    violations_metric = st.empty()
    violations_metric.metric("🚨 Active Obstructions", "0")

st.divider()

# 📐 Main UI Layout
col1, col2 = st.columns([2, 1])
with col1:
    frame_window = st.empty()
with col2:
    st.subheader("📋 Live Dispatch Alerts")
    alert_box = st.empty()
    alert_box.info("Awaiting system deployment...")

# 🛑 Core AI Logic & Analytics
BUS_LANE_POLYGON = np.array([[150, 600], [350, 300], [700, 300], [900, 600]], np.int32)

if run_system and video_path:
    try:
        model = YOLO("yolov9c.pt") 
        cap = cv2.VideoCapture(video_path)
        
        # Tracking variables
        stationary_timers = {} # Tracks how long an ID has been in the lane
        alert_logs = []
        
        while cap.isOpened() and not stop_system:
            success, frame = cap.read()
            if not success:
                st.warning("Feed ended.")
                break
                
            results = model.track(frame, classes=[2, 5, 7], conf=0.4, persist=True)
            annotated_frame = frame.copy()
            
            # Draw Clearway Zone
            cv2.polylines(annotated_frame, [BUS_LANE_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(annotated_frame, "RESTRICTED CLEARWAY", (360, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            vehicles_in_lane = 0
            violations = 0
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                total_vehicles_metric.metric("Total Vehicles Tracked", str(len(track_ids)))
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) # Center of the vehicle
                    
                    # Check if vehicle is inside the Bus Lane polygon
                    is_inside = cv2.pointPolygonTest(BUS_LANE_POLYGON, (cx, cy), False) >= 0
                    
                    color = (0, 255, 0) # Default Green
                    label = f"ID:{track_id} - Moving"
                    
                    if is_inside:
                        vehicles_in_lane += 1
                        # Increment timer for this ID
                        stationary_timers[track_id] = stationary_timers.get(track_id, 0) + 1
                        
                        # Rule: If in lane for more than 30 frames, it's an obstruction!
                        if stationary_timers[track_id] > 30: 
                            color = (0, 0, 255) # Red for Violation
                            label = f"ID:{track_id} - OBSTRUCTION"
                            violations += 1
                            if f"🚨 Vehicle {track_id} obstructing clearway!" not in alert_logs:
                                alert_logs.insert(0, f"🚨 Vehicle {track_id} obstructing clearway!")
                    else:
                        # Reset timer if they leave the lane
                        if track_id in stationary_timers:
                            del stationary_timers[track_id]

                    # Draw Box and Label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update Metrics Live
            lane_occupants_metric.metric("Vehicles in Clearway", str(vehicles_in_lane))
            violations_metric.metric("🚨 Active Obstructions", str(violations))
            
            # Update Alert Log UI
            if alert_logs:
                alert_box.error("\n\n".join(alert_logs[:5])) # Show top 5 recent alerts
                
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        
    except Exception as e:
        st.error(f"System Error: {e}")
