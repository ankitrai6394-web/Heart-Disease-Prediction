
#----------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import pickle
import joblib


# Load model files
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below")

# -------------------------
# Inputs
# -------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 40)

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        50, 250, 120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        50, 700, 200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

with col2:
    max_hr = st.number_input(
        "Maximum Heart Rate",
        60, 250, 150
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0, 10.0, 1.0
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )

rest_ecg = st.selectbox(
    "Resting ECG",
    ["LVH", "Normal", "ST"]
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["No", "Yes"]
)

st_slope = st.selectbox(
    "ST Slope",
    ["Down", "Flat", "Up"]
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Heart Disease"):

    data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_M': 1 if sex == "Male" else 0,

        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
        'ChestPainType_TA': 1 if chest_pain == "TA" else 0,

        'RestingECG_Normal': 1 if rest_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if rest_ecg == "ST" else 0,

        'ExerciseAngina_Y': 1 if exercise_angina == "Yes" else 0,

        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Up" else 0
    }

    df = pd.DataFrame([data])

    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("💔  High Risk of Heart Disease ⚠️")
        st.snow()
        st.image("https://media.giphy.com/media/l0HlPwMAzh13pcZ20/giphy.gif",width=500)
        
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.balloons()

