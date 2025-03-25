import os
import streamlit as st
from groq import Groq
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def init_services():
    return {
        'llm': Groq(api_key=os.getenv("GROQ_API_KEY")),
        'model': YOLO('yolov8s.pt')  # More general-purpose model
    }

services = init_services()

# Expanded object classes for general detection
COMMON_OBJECTS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
    7: 'truck', 16: 'dog', 17: 'cat', 56: 'chair', 67: 'cell phone'
}

def analyze_image(image):
    """General-purpose image analysis"""
    results = services['model'](image)
    detected_objects = {}
    
    for box in results[0].boxes:
        class_id = int(box.cls)
        if class_id in COMMON_OBJECTS:
            obj_name = COMMON_OBJECTS[class_id]
            detected_objects[obj_name] = detected_objects.get(obj_name, 0) + 1
    
    return detected_objects, results[0].plot()

def generate_insights(detected_objects, is_traffic=False):
    """Context-aware report generation"""
    if not detected_objects:
        return "No recognizable objects detected in the image."
    
    if is_traffic:
        prompt = f"""
        Analyze this traffic scene:
        Objects detected: {', '.join(f'{k}:{v}' for k,v in detected_objects.items())}
        Provide traffic analysis including:
        1. Vehicle density assessment
        2. Potential bottlenecks
        3. Safety observations
        """
    else:
        prompt = f"""
        Analyze this general image:
        Objects detected: {', '.join(f'{k}:{v}' for k,v in detected_objects.items())}
        Describe the scene including:
        1. Main subjects
        2. Possible setting/context
        3. Interesting observations
        """
    
    response = services['llm'].chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a versatile image analysis AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

def is_traffic_scene(detected_objects):
    """Determine if image contains traffic"""
    traffic_objects = {'car', 'bus', 'truck', 'motorcycle', 'bicycle'}
    return any(obj in traffic_objects for obj in detected_objects)

# Streamlit UI
st.title("🔍 Smart Image Analyzer")
uploaded_file = st.file_uploader("Upload any image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        objects, result_img = analyze_image(image)
        traffic = is_traffic_scene(objects)
        insights = generate_insights(objects, is_traffic=traffic)
        
        with col2:
            st.image(result_img, caption="Detected Objects", use_column_width=True)
        
        st.subheader("Analysis Report")
        st.write(insights)
        
        if objects:
            st.subheader("Detected Objects")
            st.bar_chart(objects)
        else:
            st.warning("No common objects detected")
