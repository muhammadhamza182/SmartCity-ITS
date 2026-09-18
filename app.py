import streamlit as st
import cv2
from ultralytics import YOLO

# 🎨 Page Layout & Styling
st.set_page_config(page_title="Smart City ITS", page_icon="🚦", layout="wide")

st.title("🚦 Smart City: Clearway Obstruction Detection")
st.markdown("*Real-time edge computer vision tracking system to identify traffic bottlenecks.*")

# ⚙️ Sidebar for Controls (Professional Touch)
st.sidebar.header("⚙️ System Controls")
st.sidebar.markdown("Use these controls to manage the edge device feed.")
run_system = st.sidebar.button("Start AI Detection 🚀")
stop_system = st.sidebar.button("Stop System 🛑")

# 📐 UI Layout (Left: Video, Right: Analytics)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live AI Camera Feed")
    # Yeh empty placeholder hai jahan hum frame-by-frame video chalayenge
    frame_window = st.empty()

with col2:
    st.subheader("🚨 Live Alerts & Analytics")
    status_text = st.empty()
    status_text.info("System Ready. Click 'Start AI Detection' in the sidebar.")

# 🧠 Core AI Logic
if run_system:
    status_text.warning("Loading YOLOv9 AI Model... Please wait ⏳")
    
    try:
        # 1. Load Model (YOLOv9)
        model = YOLO("yolov9c.pt") 
        status_text.success("✅ AI Model Active! Monitoring live traffic...")
        
        # 2. Open Video Stream
        cap = cv2.VideoCapture('video.mp4')
        
        # 3. Read Frame by Frame
        while cap.isOpened() and not stop_system:
            success, frame = cap.read()
            if not success:
                st.warning("Video feed ended.")
                break
            
            # 4. AI Detection (Classes: 2=Car, 5=Bus, 7=Truck)
            # conf=0.4 matlab sirf woh gariyan dikhaye jin par AI 40%+ sure hai
            results = model(frame, classes=[2, 5, 7], conf=0.4)
            
            # 5. Draw Bounding Boxes
            annotated_frame = results[0].plot()
            
            # 6. Streamlit mein show karne ke liye colors fix karna (BGR to RGB)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 7. Live video ko UI mein update karna
            frame_window.image(frame_rgb, channels="RGB", use_column_width=True)
            
        cap.release()
        
    except Exception as e:
        st.error(f"Error in processing: {e}")