import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
from datetime import datetime

# 🎨 Page Layout
st.set_page_config(page_title="Clearway AI Ops", page_icon="🚦", layout="wide")

st.title("🚦 Smart City: Edge CV Traffic Management")
st.markdown("**Optimized real-time obstruction tracking, time-based enforcement, and vehicle analytics.**")
st.divider()

# ⚙️ Sidebar Controls
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
st.sidebar.header("⏰ Enforcement Rules")
enforce_hours = st.sidebar.checkbox("Enable Time-Based Restrictions", value=True)
start_hour = st.sidebar.slider("Active Start Hour (24h)", 0, 23, 8)
end_hour = st.sidebar.slider("Active End Hour (24h)", 0, 23, 20)

st.sidebar.markdown("---")
show_heatmap = st.sidebar.checkbox("🔥 Enable Traffic Heatmap", value=False)
obstruction_threshold = st.sidebar.slider("Stationary Frames Limit", 10, 100, 25)

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
    st.subheader("📊 Vehicle Type Breakdown")
    chart_placeholder = st.empty()
    
    st.subheader("📋 Live Dispatch Alerts")
    alert_box = st.empty()
    export_box = st.empty()
    alert_box.info("Awaiting system deployment...")

BUS_LANE_POLYGON = np.array([[150, 600], [350, 300], [700, 300], [900, 600]], np.int32)

if run_system and video_path:
    try:
        model = YOLO("yolov9c.pt") 
        cap = cv2.VideoCapture(video_path)
        
        vehicle_history = {} 
        alert_logs = []      
        export_data = []     
        
        # 🔥 Fix: First frame read kar ke pehle resize karein taake heatmap layer ka size match ho jaye
        success, first_frame = cap.read()
        if success:
            first_frame = cv2.resize(first_frame, (640, 360))
            heatmap_layer = np.zeros_like(first_frame, dtype=np.uint8)
        
        # Video dobara start se parhne ke liye capture ko reset karna zaroori hai agar pehla frame parh lia ho
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened() and not stop_system:
            success, frame = cap.read()
            if not success:
                st.warning("Feed ended.")
                break
            
            # ⚡ Speed Optimization: Frame resizing
            frame = cv2.resize(frame, (640, 360))
            scaled_polygon = (BUS_LANE_POLYGON * 0.6).astype(np.int32)
            
            # Time-Based Logic Check
            current_hour = datetime.now().hour
            is_enforcement_active = True
            if enforce_hours:
                if not (start_hour <= current_hour <= end_hour):
                    is_enforcement_active = False

            results = model.track(frame, classes=[2, 5, 7], conf=0.4, persist=True)
            annotated_frame = frame.copy()
            
            zone_color = (255, 0, 0) if is_enforcement_active else (128, 128, 128)
            cv2.polylines(annotated_frame, [scaled_polygon], isClosed=True, color=zone_color, thickness=2)
            zone_text = "RESTRICTED CLEARWAY" if is_enforcement_active else "CLEARWAY (OFF-HOURS)"
            cv2.putText(annotated_frame, zone_text, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)
            
            vehicles_in_lane = 0
            violations = 0
            type_counts = {"Car": 0, "Bus": 0, "Truck": 0}
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.cpu().numpy()
                
                total_vehicles_metric.metric("Total Vehicles Tracked", str(len(track_ids)))
                
                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) 
                    
                    if cls_id == 2: type_counts["Car"] += 1
                    elif cls_id == 5: type_counts["Bus"] += 1
                    elif cls_id == 7: type_counts["Truck"] += 1
                    
                    if show_heatmap:
                        cv2.circle(heatmap_layer, (cx, cy), 10, (0, 0, 255), -1)
                    
                    is_inside = cv2.pointPolygonTest(scaled_polygon, (cx, cy), False) >= 0
                    color = (0, 255, 0) 
                    label = f"ID:{track_id}"
                    
                    if track_id in vehicle_history:
                        last_cx, last_cy, stat_frames = vehicle_history[track_id]
                        distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                        if distance < 3: 
                            stat_frames += 1
                        else:
                            stat_frames = 0 
                    else:
                        stat_frames = 0
                        
                    vehicle_history[track_id] = (cx, cy, stat_frames)

                    if is_inside and is_enforcement_active:
                        vehicles_in_lane += 1
                        if stat_frames > obstruction_threshold: 
                            color = (0, 0, 255) 
                            label = f"ID:{track_id} OBSTRUCTING!"
                            violations += 1
                            
                            time_now = datetime.now().strftime("%H:%M:%S")
                            log_msg = f"🚨 [{time_now}] Vehicle {track_id} blocking lane!"
                            
                            if len(alert_logs) == 0 or log_msg not in alert_logs[0]:
                                alert_logs.insert(0, log_msg)
                                export_data.append({"Time": time_now, "Vehicle ID": track_id, "Status": "Obstruction"})

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

            if show_heatmap:
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_layer, 0.3, 0)

            lane_occupants_metric.metric("Vehicles in Clearway", str(vehicles_in_lane))
            violations_metric.metric("🚨 Active Obstructions", str(violations))
            
            df_chart = pd.DataFrame(list(type_counts.items()), columns=["Vehicle Type", "Count"])
            chart_placeholder.bar_chart(df_chart.set_index("Vehicle Type"))
            
            if alert_logs:
                alert_box.error("\n\n".join(alert_logs[:4])) 
                
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        
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
