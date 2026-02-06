import streamlit as st
from ultralytics import YOLO
import numpy as np
import time
import pandas as pd
from PIL import Image

# Page config
st.set_page_config(page_title="YOLOv8 Image Detection", layout="wide")

st.title("🔍 YOLOv8 Image Detection Demo")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("models/yolov8s_best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Detect"):
        start_time = time.time()

        results = model.predict(
            source=image_np,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        end_time = time.time()
        inference_time_ms = int((end_time - start_time) * 1000)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                detections.append({
                    "Label": label,
                    "Confidence": round(conf, 3)
                })

            annotated_img = r.plot()

        with col2:
            st.subheader("Annotated Image")
            st.image(annotated_img, use_container_width=True)

        st.markdown("---")

        st.metric("Inference Time (ms)", inference_time_ms)

        if detections:
            df = pd.DataFrame(detections)
            st.subheader("Detections")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No objects detected.")
