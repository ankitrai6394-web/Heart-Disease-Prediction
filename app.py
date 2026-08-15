import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* ================================
       MAIN APP BACKGROUND
       ================================ */

    .stApp {
        background: linear-gradient(
            135deg,
            #eaf4ff 0%,
            #ffffff 50%,
            #eef8f5 100%
        );
        color: #172033;
    }


    /* ================================
       MAIN CONTENT
       ================================ */

    .main {
        color: #172033;
    }


    /* ================================
       TITLE
       ================================ */

    h1 {
        color: #12355b !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #174a7c !important;
    }


    /* ================================
       NORMAL TEXT
       ================================ */

    p {
        color: #26364a !important;
        font-size: 17px;
    }


    /* ================================
       INPUT LABELS
       ================================ */

    label {
        color: #17324d !important;
        font-weight: 600 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #17324d !important;
        font-weight: 600 !important;
    }


    /* ================================
       NUMBER INPUT
       ================================ */

    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border-radius: 8px;
    }

    div[data-baseweb="input"] input {
        color: #172033 !important;
        background-color: #ffffff !important;
        font-size: 16px !important;
    }


    /* ================================
       SELECT BOX
       ================================ */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-radius: 8px;
    }

    div[data-baseweb="select"] span {
        color: #172033 !important;
    }


    /* ================================
       SELECTBOX DROPDOWN
       ================================ */

    ul[role="listbox"] {
        background-color: #ffffff !important;
    }

    li[role="option"] {
        color: #172033 !important;
        background-color: #ffffff !important;
    }

    li[role="option"]:hover {
        background-color: #eaf4ff !important;
        color: #12355b !important;
    }


    /* ================================
       BUTTON
       ================================ */

    .stButton > button {
        background-color: #1769aa !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 25px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #0d4f85 !important;
        color: white !important;
    }


    /* ================================
       SUCCESS MESSAGE
       ================================ */

    div[data-testid="stAlert"] {
        border-radius: 10px;
        font-weight: 600;
    }


    /* ================================
       ERROR MESSAGE
       ================================ */

    div[data-testid="stAlert"] p {
        font-weight: 700 !important;
    }


    /* ================================
       COLUMN SPACING
       ================================ */

    div[data-testid="column"] {
        padding: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below")


# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        18,
        100,
        40
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        50,
        250,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        50,
        700,
        200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )


with col2:

    max_hr = st.number_input(
        "Maximum Heart Rate",
        60,
        250,
        150
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )


# ---------------------------------------------------------
# Additional Inputs
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict Heart Disease"):

    data = {

        'Age': age,

        'RestingBP': resting_bp,

        'Cholesterol': cholesterol,

        'FastingBS': fasting_bs,

        'MaxHR': max_hr,

        'Oldpeak': oldpeak,

        'Sex_M': 1 if sex == "Male" else 0,

        'ChestPainType_ATA':
            1 if chest_pain == "ATA" else 0,

        'ChestPainType_NAP':
            1 if chest_pain == "NAP" else 0,

        'ChestPainType_TA':
            1 if chest_pain == "TA" else 0,

        'RestingECG_Normal':
            1 if rest_ecg == "Normal" else 0,

        'RestingECG_ST':
            1 if rest_ecg == "ST" else 0,

        'ExerciseAngina_Y':
            1 if exercise_angina == "Yes" else 0,

        'ST_Slope_Flat':
            1 if st_slope == "Flat" else 0,

        'ST_Slope_Up':
            1 if st_slope == "Up" else 0
    }


    # Convert to DataFrame

    df = pd.DataFrame([data])


    # Scale input

    scaled_data = scaler.transform(df)


    # Prediction

    prediction = model.predict(scaled_data)[0]


    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    if prediction == 1:

        st.error(
            "💔 High Risk of Heart Disease ⚠️"
        )

        st.markdown(
            "## 🚨⚠️🚨⚠️🚨⚠️🚨"
        )

        st.snow()


    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

        st.balloons()
