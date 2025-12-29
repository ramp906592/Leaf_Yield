# 🌱 Smart Crop Disease & Yield Prediction System

An AI-powered agriculture support system that helps farmers and agricultural experts detect crop diseases from leaf images and predict crop yield using environmental and farming data.

## 📌 Project Overview

This project integrates **Deep Learning (CNN)** and **Machine Learning (ANN)** models into a single user-friendly web application to support smart agricultural decision-making.

### 🔹 Key Features
- 🌿 **Crop Disease Detection** using leaf images  
- 📈 **Crop Yield Prediction** based on climate and farming inputs  
- 🧠 AI-driven insights to reduce crop loss and improve productivity  
- 🖥️ Interactive **Streamlit web interface**

## 🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **NumPy & Pandas**
- **Streamlit**
- **Git & GitHub (Git LFS)**


## 🧪 Models Used

### 1️⃣ Crop Disease Detection (CNN)
- Trained on plant leaf images (PlantVillage-style dataset)
- Supports **38 crop disease and healthy classes**
- Input: Leaf image  
- Output: Disease name + confidence score  

### 2️⃣ Crop Yield Prediction (ANN)
- Trained on historical crop yield and climate data
- Input features:
  - Area / Country
  - Crop type
  - Year
  - Average rainfall
  - Pesticides usage
  - Average temperature
- Output:  
  - Predicted crop yield (quintal/hectare & tons/hectare)


## 🖥️ Web Application Pages

- 🏠 **Home** – Project overview and usage guide  
- 🌿 **Crop Disease Detection** – Upload leaf image for disease prediction  
- 📈 **Crop Yield Prediction** – Enter environmental details for yield estimation  
- ℹ️ **About** – Project and technical details  

