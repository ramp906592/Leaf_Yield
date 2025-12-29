import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Crop Disease & Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# CUSTOM CSS & STYLING (Nature/Glassmorphism)
# =====================================================
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(to right, #0F2027, #203A43, #2C5364); /* Deep Nature Gradient */
        color: white;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.9); /* Glassy Dark Sidebar */
        backdrop-filter: blur(10px);
    }
    
    /* Card Styles - Glassmorphism */
    .card {
        background: rgba(255, 255, 255, 0.05); /* Ultra transparent white */
        backdrop-filter: blur(12px); /* Frost effect */
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 20px;
        padding: 30px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .card h3 {
        margin-bottom: 15px;
        font-weight: 700;
        color: #A8E6CF; /* Soft Green Title */
    }
    .card p {
        font-size: 1.15rem;
        margin-bottom: 25px;
        color: #dcdde1;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Button Styling - Lush Green */
    .stButton > button {
        background: linear-gradient(135deg, #11998e, #38ef7d); /* Lush Green Gradient */
        border: none;
        color: white;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 17px;
        font-weight: 600;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 30px; /* Pill shaped */
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0e857a, #2ecc71);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.5);
    }
    
    /* Adjust headings color */
    h1, h2, h3 {
        color: #A8E6CF !important; /* Soft Green Headings */
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE NAVIGATION
# =====================================================
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

# =====================================================
# SCALER VALUES (FROM TRAINING)
# Feature order:
# [Year, Rainfall, Pesticides, Temperature]
# =====================================================
SCALER_MEAN = np.array([
    2.00156863e+03,
    1.14402452e+03,
    3.71427680e+04,
    2.05065489e+01
])

SCALER_SCALE = np.array([
    7.05651147e+00,
    7.07271096e+02,
    6.00480554e+04,
    6.34043542e+00
])

# =====================================================
# DISEASE CLASS NAMES (38 CLASSES)
# =====================================================
DISEASE_CLASSES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora Leaf Spot", "Corn Common Rust",
    "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Huanglongbing (Citrus Greening)",
    "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight",
    "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus", "Tomato Healthy"
]

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    cnn = load_model("crop_disease_cnn.h5", compile=False)
    ann = load_model("crop_yield_annnn_model.h5", compile=False)
    return cnn, ann

cnn_model, ann_model = load_models()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌱 Navigation")

# Sync sidebar with session state
# Ensure session state page is valid
valid_pages = ["Home", "Crop Yield Prediction", "Leaf Disease Detection"]
if st.session_state.page not in valid_pages:
    st.session_state.page = "Home"

page = st.sidebar.radio(
    "Select Page",
    valid_pages,
    index=valid_pages.index(st.session_state.page),
    key="navigation_radio"
)

# Update session state if sidebar is used
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

# =====================================================
# HOME
# =====================================================
if page == "Home":
    st.title("🌱 Smart Crop Disease & Yield Prediction System")
    
    st.markdown("### Welcome! Choose a service below:", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🍃 Leaf Disease Detection</h3>
            <p>Upload a photo of a crop leaf to detect diseases using our advanced CNN model.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Disease Detection 🍃", use_container_width=True):
            st.session_state.page = "Leaf Disease Detection"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <h3>🌾 Crop Yield Prediction</h3>
            <p>Predict crop yield based on environmental factors like rainfall, temperature, and more.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Yield Prediction 🌾", use_container_width=True):
             st.session_state.page = "Crop Yield Prediction"
             st.rerun()

# =====================================================
# CROP YIELD PREDICTION
# =====================================================
elif page == "Crop Yield Prediction":
    st.title("🌾 Crop Yield Prediction")

    year = st.number_input("Year", min_value=1990, max_value=2035, value=2024)
    rainfall = st.number_input("Average Rainfall (mm/year)", value=1200.0)
    pesticides = st.number_input("Pesticides Used (tonnes)", value=100.0)
    temperature = st.number_input("Average Temperature (°C)", value=25.0)

    if st.button("Predict Yield"):

        # 🔧 FIX 1: Prevent ANN extrapolation
        if year > 2013:
            year = 2013

        X_input = np.array([[year, rainfall, pesticides, temperature]])
        X_scaled = (X_input - SCALER_MEAN) / SCALER_SCALE

        pred = ann_model.predict(X_scaled)
        yield_pred = float(pred[0][0])

        # 🔧 FIX 2: Cap unrealistic predictions
        yield_pred = max(0, min(yield_pred, 80000))

        st.success(f"🌾 Predicted Yield: {yield_pred:.2f} hg/ha")
        st.info(f"📦 Approx: {yield_pred / 1000:.2f} tons/hectare")

        st.caption("ℹ️ Yield is estimated based on historical climate patterns.")

# =====================================================
# LEAF DISEASE DETECTION
# =====================================================
elif page == "Leaf Disease Detection":
    st.title("🍃 Leaf Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", width=250)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Predict Disease"):
            preds = cnn_model.predict(img_array)
            class_index = int(np.argmax(preds))
            confidence = float(np.max(preds)) * 100
            disease_name = DISEASE_CLASSES[class_index]

            st.success(f"🦠 Disease Detected: {disease_name}")
            st.info(f"🔍 Confidence: {confidence:.2f}%")
