import os
import streamlit as st
from groq import Groq
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize services
@st.cache_resource
def init_services():
    return {
        'llm': Groq(api_key=os.getenv("GROQ_API_KEY")),
        'model': YOLO('yolov8s.pt')  # More accurate model
    }

services = init_services()

# Enhanced object categories
OBJECT_CATEGORIES = {
    'vehicles': ['car', 'bus', 'truck', 'motorcycle', 'bicycle'],
    'people': ['person'],
    'animals': ['dog', 'cat', 'bird'],
    'furniture': ['chair', 'table', 'couch'],
    'electronics': ['cell phone', 'laptop', 'tv'],
    'food': ['banana', 'apple', 'sandwich', 'pizza']
}

def analyze_image(image):
    """Comprehensive image analysis with enhanced detection"""
    results = services['model'](image)
    
    # Categorize detected objects
    object_counts = defaultdict(int)
    detected_categories = defaultdict(list)
    
    for box in results[0].boxes:
        class_name = services['model'].names[int(box.cls)]
        object_counts[class_name] += 1
        
        # Categorize the object
        for category, items in OBJECT_CATEGORIES.items():
            if class_name in items:
                detected_categories[category].append(class_name)
    
    # Get scene composition
    width, height = image.size
    dominant_color = get_dominant_color(image) if width < 2000 else None
    
    return {
        'objects': dict(object_counts),
        'categories': dict(detected_categories),
        'composition': {
            'width': width,
            'height': height,
            'aspect_ratio': round(width/height, 2),
            'dominant_color': dominant_color
        },
        'annotated_image': results[0].plot()
    }

def get_dominant_color(image, k=1):
    """Get dominant color (simplified version)"""
    img = image.resize((100, 100)).convert('RGB')
    pixels = list(img.getdata())
    return max(set(pixels), key=pixels.count) if pixels else None

def generate_insights(analysis):
    """Generate comprehensive image report"""
    # Prepare context
    context = {
        'total_objects': sum(analysis['objects'].values()),
        'object_diversity': len(analysis['objects']),
        'primary_categories': [k for k,v in analysis['categories'].items() if v],
        'composition': analysis['composition']
    }
    
    # Generate appropriate prompt
    if context['total_objects'] == 0:
        return "No objects detected. This might be an abstract image or poor quality photo."
    
    prompt = f"""
    Analyze this image based on:
    
    **Objects Detected**:
    - Total: {context['total_objects']}
    - Unique: {context['object_diversity']}
    - By Category: {', '.join(f"{k}({len(v)})" for k,v in analysis['categories'].items() if v)}
    
    **Image Composition**:
    - Resolution: {context['composition']['width']}x{context['composition']['height']}
    - Aspect Ratio: {context['composition']['aspect_ratio']}
    - Dominant Color: {context['composition']['dominant_color'] or 'N/A'}
    
    Provide a detailed report including:
    1. Probable setting (indoor/outdoor, environment type)
    2. Key observations about objects and their relationships
    3. Interesting compositional notes
    4. Any unusual or notable elements
    """
    
    response = services['llm'].chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an expert image analyst with knowledge in photography, visual arts, and scene understanding."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(layout="wide", page_title="Smart Image Analyzer")
st.title(" Smart Image Analyzer")

uploaded_file = st.file_uploader(
    "Upload any image for analysis", 
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

if uploaded_file:
    image = Image.open(uploaded_file)
    
    with st.spinner("Analyzing image..."):
        analysis = analyze_image(image)
        insights = generate_insights(analysis)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(analysis['annotated_image'], caption="Detected Objects", use_container_width=True)
    
    # Insights section
    st.subheader("Comprehensive Analysis")
    st.write(insights)
    
    # Data sections
    with st.expander(" Object Statistics", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            st.write("**By Quantity**")
            st.bar_chart(analysis['objects'])
        with col4:
            st.write("**By Category**")
            st.json({k: len(v) for k,v in analysis['categories'].items() if v})
    
    with st.expander("Composition Analysis"):
        st.json(analysis['composition'])
