import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionAI Pro", layout="wide", page_icon="🔍")

# --- UI STYLING (Transparent Tech Aesthetic) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020617);
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism Cards for Metrics and Containers */
    div[data-testid="stMetric"], .stImage, .stFileUploader {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 150, 255, 0.2);
    }

    /* Massive Neon Title */
    .main-title {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 500px;
        font-weight: 900;
        text-align: center;
      
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.5));
    }

    .sub-title {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 40px;
    }

    /* Customizing Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1e40af, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.6) !important;
        transform: scale(1.02);
    }

    /* Remove standard Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Text Input/Selectbox adjustments */
    .stSelectbox label, .stSlider label {
        color: #60a5fa !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    # Using small model for speed, change to 'yolov8m.pt' for better accuracy
    return YOLO("yolov8n.pt")

with st.spinner("⚡ Initializing Neural Core..."):
    model = load_model()

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown('<p class="main-title">VISION AI PRO</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title"> Object Detection</p>', unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:
    #st.markdown("## 🛠️ System Config")
    source = st.selectbox(
        "Input Protocol",
        ["Image Upload", "Live Webcam"],
        index=0
    )
    
    confidence = st.slider(
        "Detection Sensitivity",
        min_value=0.0, max_value=1.0, value=0.45, step=0.05
    )
    
    st.divider()
    #st.info("System Status: Online 🟢")

# ---------------------------------
# MAIN INTERFACE
# ---------------------------------
col1, col2 = st.columns([1, 1], gap="large")

if source == "Image Upload":
    with col1:
        st.markdown("###  Data Input")
        uploaded_file = st.file_uploader("Drop image here", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Raw Input Stream", use_container_width=True)

    with col2:
        st.markdown("###  Image Analysis")
        if uploaded_file:
            start_time = time.time()
            results = model(img, conf=confidence)
            end_time = time.time()
            
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Processed Neural Output", use_container_width=True)
            
            m1, m2 = st.columns(2)
            m1.metric("Objects Identified", len(results[0].boxes))
            m2.metric("Processing Latency", f"{int((end_time - start_time) * 1000)}ms")
            
            # Export
            result_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button(
                label=" Export Processed Data",
                data=buf.getvalue(),
                file_name="vision_analysis.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("Awaiting input data for analysis...")

elif source == "Live Webcam":
    with col1:
        st.markdown("###  Webcam")
        run_cam = st.toggle("Turn on Camera", value=False)
        #st.write("Real-time frame-by-frame processing enabled when active.")

    with col2:
        st.markdown("###  Live Stream")
        FRAME_WINDOW = st.image([])
        
        if run_cam:
            camera = cv2.VideoCapture(0)
            while run_cam:
                ret, frame = camera.read()
                if not ret: break
                
                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()
                
                FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            camera.release()
       # else:
            #st.info("Camera offline. Toggle 'Neural Link' to begin.")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Quantum-Ready Computer Vision Pipeline • v2.4.0 • © 2026")