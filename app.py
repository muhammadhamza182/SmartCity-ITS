import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
from datetime import datetime

# 🎨 Premium Page Layout (Clean & Structured)
st.set_page_config(page_title="Clearway AI Ops", page_icon="🚦", layout="wide")

# 🪧 Header Section
st.title("🚦 Smart City: Edge CV Traffic Management")
st.markdown("**Real-time obstruction tracking, heatmaps, and automated dispatch for intelligent transportation.**")
st.divider()

# ⚙️ Sidebar: Dynamic Controls & Features
st.sidebar.header("⚙️ Data Source")
source_type = st.sidebar.radio("Select Feed Type:", ["Upload Video", "Live Camera (RTSP)"])

video_path = None
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload CCTV Footage (MP4)", type=['mp4'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
else:
    video_path = st.sidebar.text_input("Enter RTSP Stream Link:", "rtsp://username:password@ip_address/stream")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Advanced Features")
show_heatmap = st.sidebar.checkbox("🔥 Enable Traffic Heatmap", value=False)
obstruction_threshold = st.sidebar.slider("Stationary Time (Frames)", 10, 100, 30)

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
    export_box = st.empty()
    alert_box.info("Awaiting system deployment...")

# 🛑 Core AI & Spatial Logic
BUS_LANE_POLYGON = np.array([[150, 600], [350, 300], [700, 300], [900, 600]], np.int32)

if run_system and video_path:
    try:
        model = YOLO("yolov9c.pt") 
        cap = cv2.VideoCapture(video_path)
        
        # 🧠 Smart Tracking Variables
        vehicle_history = {} # {id: (last_cx, last_cy, stationary_frames)}
        alert_logs = []      # For UI Display
        export_data = []     # For CSV Export
        
        # 🔥 Heatmap Layer Initialization
        success, first_frame = cap.read()
        if success:
            heatmap_layer = np.zeros_like(first_frame, dtype=np.uint8)
        
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
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) 
                    
                    # 🔥 Draw Heatmap points
                    if show_heatmap:
                        cv2.circle(heatmap_layer, (cx, cy), 15, (0, 0, 255), -1)
                    
                    is_inside = cv2.pointPolygonTest(BUS_LANE_POLYGON, (cx, cy), False) >= 0
                    color = (0, 255, 0) # Green (Moving)
                    label = f"ID:{track_id} - Moving"
                    
                    # 🧠 True Stationary Logic (Pixel Distance Calculation)
                    if track_id in vehicle_history:
                        last_cx, last_cy, stat_frames = vehicle_history[track_id]
                        distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                        
                        if distance < 5: # Agar gari 5 pixel se kam move hui (yani ruki hui hai)
                            stat_frames += 1
                        else:
                            stat_frames = 0 # Agar chal pari toh timer reset
                    else:
                        stat_frames = 0
                        
                    vehicle_history[track_id] = (cx, cy, stat_frames)

                    # Alert Logic
                    if is_inside:
                        vehicles_in_lane += 1
                        if stat_frames > obstruction_threshold: 
                            color = (0, 0, 255) # Red (Violation)
                            label = f"ID:{track_id} - OBSTRUCTION"
                            violations += 1
                            
                            time_now = datetime.now().strftime("%H:%M:%S")
                            log_msg = f"🚨 [{time_now}] Vehicle {track_id} obstructing clearway!"
                            
                            if len(alert_logs) == 0 or log_msg not in alert_logs[0]:
                                alert_logs.insert(0, log_msg)
                                export_data.append({"Time": time_now, "Vehicle ID": track_id, "Status": "Obstruction"})

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 🔥 Blend Heatmap with Main Frame
            if show_heatmap:
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_layer, 0.3, 0)

            # Update Metrics Live
            lane_occupants_metric.metric("Vehicles in Clearway", str(vehicles_in_lane))
            violations_metric.metric("🚨 Active Obstructions", str(violations))
            
            # Update Alert Logs
            if alert_logs:
                alert_box.error("\n\n".join(alert_logs[:5])) 
                
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        
        # 📥 CSV Export Feature (Jahan video khatam ho ya stop ho)
        if export_data:
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False).encode('utf-8')
            with export_box:
                st.download_button(
                    label="📥 Download Violation Report (CSV)",
                    data=csv,
                    file_name="clearway_violations.csv",
                    mime="text/csv",
                )
        
    except Exception as e:
        st.error(f"System Error: {e}")
