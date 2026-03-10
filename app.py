# app.py - Face Recognition with working webcam (no complex dependencies)

import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import os
import tempfile
from datetime import datetime
import pandas as pd
import face_recognition

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👥",
    layout="wide"
)

# ============================================
# LOAD YOUR TRAINED MODEL
# ============================================
MODEL_PATH = 'face_recognizer_20260307_070311.pkl'

class SimpleFaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_types = []

    def recognize(self, image_path, tolerance=0.5):
        """
        REAL face recognition using your trained model
        """
        try:
            # Load image
            unknown_image = face_recognition.load_image_file(image_path)
            
            # Find faces
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
            
            if len(face_encodings) == 0:
                return "NO_FACE", 0.0, "No face detected"
            
            if len(self.known_face_encodings) == 0:
                return "NO_KNOWN_FACES", 0.0, "No faces in database"
            
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Convert distance to confidence
            confidence = 1 - min(best_distance, 1.0)
            
            if best_distance < tolerance:
                name = self.known_face_names[best_match_index]
                face_type = self.known_face_types[best_match_index] if best_match_index < len(self.known_face_types) else 'team'
                
                if face_type == 'team':
                    result = f"TEAM_MEMBER_{name}"
                else:
                    result = f"OTHER_PERSON_{name}"
            else:
                result = "UNKNOWN_PERSON"
            
            return result, confidence, f"Match found with distance: {best_distance:.3f}"
            
        except Exception as e:
            return "ERROR", 0.0, str(e)

# Load model
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

recognizer = load_model()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("👥 Face Recognition")
    st.markdown("---")
    
    if recognizer and hasattr(recognizer, 'known_face_names'):
        st.success("✅ Model Loaded")
        st.metric("Total Faces", len(recognizer.known_face_encodings))
        
        # Show team members
        st.subheader("👥 Team Members")
        for i, name in enumerate(recognizer.known_face_names):
            if i < len(recognizer.known_face_types) and recognizer.known_face_types[i] == 'team':
                st.write(f"• {name}")
        
        # Recognition settings
        st.markdown("---")
        st.subheader("⚙️ Settings")
        tolerance = st.slider(
            "Recognition Tolerance",
            min_value=0.3,
            max_value=0.7,
            value=0.5,
            step=0.05,
            help="Lower = stricter matching"
        )
    else:
        st.error("❌ Model not loaded")

# ============================================
# MAIN TABS
# ============================================
tab1, tab2 = st.tabs(["📷 Upload Image", "📸 Webcam Capture"])

# ============================================
# TAB 1: UPLOAD IMAGE
# ============================================
with tab1:
    st.header("Upload Image for Recognition")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 RECOGNIZE", type="primary", use_container_width=True):
                with st.spinner("Comparing with database..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    # Recognize
                    result, confidence, message = recognizer.recognize(temp_path, tolerance)
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    # Show result
                    st.subheader("📊 Result")
                    
                    if result == "NO_FACE":
                        st.error("❌ No face detected")
                    elif result.startswith('TEAM_MEMBER_'):
                        member_name = result.replace('TEAM_MEMBER_', '')
                        st.success(f"✅ Team Member: {member_name}")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.progress(confidence)
                    elif result == "UNKNOWN_PERSON":
                        st.warning("⚠️ Unknown Person")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.progress(confidence)
                    else:
                        st.info(f"Result: {result}")

# ============================================
# TAB 2: WEBCAM CAPTURE (SIMPLE, NO COMPLEX DEPS)
# ============================================
with tab2:
    st.header("📸 Webcam Capture")
    st.info("Take a photo with your webcam for recognition")
    
    # Use Streamlit's built-in camera input
    img_file_buffer = st.camera_input("Take a photo")
    
    if img_file_buffer is not None:
        # Convert to PIL Image
        image = Image.open(img_file_buffer)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Captured Photo", use_container_width=True)
        
        with col2:
            if st.button("🔍 RECOGNIZE CAPTURED PHOTO", type="primary", use_container_width=True):
                with st.spinner("Recognizing..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    # Recognize
                    result, confidence, message = recognizer.recognize(temp_path, tolerance)
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    # Show result
                    st.subheader("📊 Result")
                    
                    if result == "NO_FACE":
                        st.error("❌ No face detected")
                    elif result.startswith('TEAM_MEMBER_'):
                        member_name = result.replace('TEAM_MEMBER_', '')
                        st.success(f"✅ Team Member: {member_name}")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.progress(confidence)
                    elif result == "UNKNOWN_PERSON":
                        st.warning("⚠️ Unknown Person")
                        st.metric("Confidence", f"{confidence:.2%}")
                        st.progress(confidence)
                    else:
                        st.info(f"Result: {result}")

# ============================================
# DATABASE VIEW
# ============================================
with st.expander("📊 View Database", expanded=False):
    if recognizer and hasattr(recognizer, 'known_face_names'):
        # Create dataframe
        data = []
        for i in range(len(recognizer.known_face_names)):
            face_type = recognizer.known_face_types[i] if i < len(recognizer.known_face_types) else 'unknown'
            data.append({
                "Name": recognizer.known_face_names[i],
                "Type": face_type.upper(),
                "Encoding": f"Vector {i+1}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Faces", len(recognizer.known_face_encodings))
        with col2:
            team_count = sum(1 for t in recognizer.known_face_types if t == 'team')
            st.metric("Team Members", team_count)
        with col3:
            other_count = sum(1 for t in recognizer.known_face_types if t == 'other')
            st.metric("Other Persons", other_count)