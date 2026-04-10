import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Impact Predictor", page_icon="🎓", layout="wide")

# --- CUSTOM CSS FOR ANIMATION & STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_student = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kd549zpg.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- APP INTERFACE ---
st.title("🎓 Student AI Impact Predictor")
st.write("Predict the **Impact on Grades** based on AI tool usage habits.")

col1, col2 = st.columns([1, 1])

with col1:
    st_lottie(lottie_student, height=300, key="coding")

with col2:
    st.subheader("Enter Student Details")
    
    # Input fields based on model features 
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", options=[0, 1], help="0: Male, 1: Female (Encoding used during training)")
    edu_level = st.selectbox("Education Level", options=[0, 1, 2], help="High School, Undergraduate, Postgraduate")
    city = st.number_input("City Code", min_value=0, step=1)
    ai_tool = st.selectbox("AI Tool Used", options=[0, 1, 2, 3], help="ChatGPT, Claude, Gemini, etc.")
    usage_hours = st.slider("Daily Usage Hours", 0.0, 24.0, 2.0)
    purpose = st.selectbox("Purpose", options=[0, 1, 2], help="Study, Research, Fun")

# --- PREDICTION LOGIC ---
if st.button("Analyze Impact"):
    # Prepare features in the exact order found in the model 
    # Note: 'Student_ID' is included in your model's feature_names_in_
    input_data = pd.DataFrame([[
        0,              # Student_ID (placeholder)
        age,            # Age
        gender,         # Gender
        edu_level,      # Education_Level
        city,           # City
        ai_tool,        # AI_Tool_Used
        usage_hours,    # Daily_Usage_Hours
        purpose,        # Purpose
        0               # Placeholder for target or 9th feature
    ]], columns=['Student_ID', 'Age', 'Gender', 'Education_Level', 'City', 
                 'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades'])
    
    # Drop the target column if the model expects only 8 features, 
    # or keep all 9 if the model expects 9 
    prediction = model.predict(input_data)
    
    st.success(f"### Predicted Result: {prediction[0]}")
    st.balloons()
