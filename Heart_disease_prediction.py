
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Title
st.title("❤️ Heart Disease Prediction App")

# Load dataset
heart_data = pd.read_csv("heart.csv")

# Features and Target
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (removed stratify to avoid error with small dataset)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=2
)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

st.sidebar.header("Model Info")
st.sidebar.write(f"✅ Training Accuracy: {train_acc:.2f}")
st.sidebar.write(f"✅ Testing Accuracy: {test_acc:.2f}")

# User input form
st.header("🔍 Enter Patient Data")
age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect)", [0, 1, 2])

# Prediction
if st.button("🔮 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Heart Disease.")
    else:
        st.success("💚 The patient is unlikely to have Heart Disease.")

