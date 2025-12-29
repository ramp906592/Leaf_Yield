import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import joblib
import warnings
import os

# ===============================
# CLEAN WARNINGS
# ===============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ===============================
# PAGE CONFIG
# ===============================
# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Smart Crop Disease & Yield Prediction System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS & STYLING
# ===============================
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

# ===============================
# LOAD MODELS + SCALER
# ===============================
@st.cache_resource
def load_all():
    cnn = load_model("crop_disease_cnn.h5", compile=False)
    ann = load_model("crop_yield_ann_model.h5", compile=False)
    scaler = joblib.load("yield_scaler.pkl")
    return cnn, ann, scaler

cnn_model, ann_model, scaler = load_all()

# ===============================
# CLASS NAMES (38 CLASSES)
# ===============================
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca',
    'Grape___Leaf_blight',
    'Grape___healthy',
    'Orange___Haunglongbing',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper___Bacterial_spot',
    'Pepper___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ===============================
# IMAGE PREPROCESS
# ===============================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# SESSION STATE NAVIGATION
# ===============================
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

def set_page(page_name):
    st.session_state.page = page_name

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("🌾 Navigation")

# REVISED SIDEBAR:
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🌿 Crop Disease Detection", "📈 Crop Yield Prediction"],
    key="navigation_radio",
    index=["🏠 Home", "🌿 Crop Disease Detection", "📈 Crop Yield Prediction"].index(st.session_state.page)
)

# Update session state if sidebar is used
if page != st.session_state.page:
    st.session_state.page = page


# ===============================
# 🏠 HOME
# ===============================
if page == "🏠 Home":
    st.title("🌱 Smart Crop Disease & Yield Prediction System")
    
    st.markdown("### Welcome! Choose a service below:", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🌿 Crop Disease Detection</h3>
            <p>Upload a photo of a crop leaf to detect diseases using our advanced CNN model.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Disease Detection 🌿", use_container_width=True):
            st.session_state.page = "🌿 Crop Disease Detection"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <h3>📈 Crop Yield Prediction</h3>
            <p>Predict crop yield based on environmental factors like rainfall, temperature, and more.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Yield Prediction 📈", use_container_width=True):
             st.session_state.page = "📈 Crop Yield Prediction"
             st.rerun()

    st.markdown("---")
    st.info("👈 You can also use the sidebar for navigation.")


# ======================================================
# 🌿 CROP DISEASE DETECTION
# ======================================================
elif page == "🌿 Crop Disease Detection":
    st.title("🌿 Crop Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Crop Leaf Image 👨‍🌾",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Leaf Image", width=250)

        if st.button("Predict Disease"):
            img_array = preprocess_image(img)
            preds = cnn_model.predict(img_array)
            class_index = np.argmax(preds)
            confidence = np.max(preds) * 100

            disease_name = CLASS_NAMES[class_index]

            st.balloons() # Animation on success
            st.success(f"🦠 Disease Detected: {disease_name}")
            st.info(f"🔍 Confidence: {confidence:.2f}%")

# ======================================================
# 📈 CROP YIELD PREDICTION
# ======================================================
elif page == "📈 Crop Yield Prediction":
    st.title("📈 Crop Yield Prediction")

    area = st.selectbox("Area / Country", ["Albania", "India", "USA", "Other"])
    crop = st.selectbox("Crop Type", ["Maize", "Rice", "Potatoes", "Wheat", "Soybeans"])
    year = st.number_input("Year", min_value=1990, max_value=2030, value=2024)

    rainfall = st.number_input("Average Rainfall (mm/year)", value=1200.0)
    pesticides = st.number_input("Pesticides Used (tonnes)", value=100.0)
    temperature = st.number_input("Average Temperature (°C)", value=25.0)

    if st.button("Predict Yield"):
        input_df = pd.DataFrame({
            "Area": [area],
            "Item": [crop],
            "Year": [year],
            "average_rain_fall_mm_per_year": [rainfall],
            "pesticides_tonnes": [pesticides],
            "avg_temp": [temperature]
        })

        input_df = pd.get_dummies(input_df)
        input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        required_features = ann_model.input_shape[1]
        if input_df.shape[1] < required_features:
            for i in range(required_features - input_df.shape[1]):
                input_df[f"dummy_{i}"] = 0

        X_input = input_df.iloc[:, :required_features].values.astype("float32")
        X_input = scaler.transform(X_input)

        yield_pred = ann_model.predict(X_input)[0][0]

        quintal = yield_pred / 10000
        tons = yield_pred / 100000

        st.balloons() # Animation on success
        st.success(f"🌾 Predicted Yield: {quintal:.2f} quintal/hectare")
        st.info(f"📦 Approx: {tons:.2f} tons/hectare")

st.markdown("---")
st.caption("CNN + ANN based Smart Agriculture System")
