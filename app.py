import streamlit as st
import pandas as pd
import joblib
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

try:
    with open("medical_bg.png", "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

except FileNotFoundError:
    encoded_image = ""


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
<style>

/* =======================================================
   MAIN BACKGROUND
   ======================================================= */

.stApp {{
    background-image:
        linear-gradient(
            rgba(235, 246, 255, 0.82),
            rgba(248, 252, 255, 0.93)
        ),
        url("data:image/png;base64,{encoded_image}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;

    min-height: 100vh;
}}


/* =======================================================
   MAIN CONTENT
   ======================================================= */

.block-container {{
    max-width: 1200px;

    padding-top: 2rem;
    padding-bottom: 3rem;
}}


/* =======================================================
   HEADER
   ======================================================= */

h1 {{
    color: #123b66 !important;

    text-align: center;

    font-size: 42px !important;

    font-weight: 800 !important;

    margin-bottom: 5px !important;

    letter-spacing: -0.5px;
}}


/* =======================================================
   SUBTITLE
   ======================================================= */

.subtitle {{
    text-align: center;

    color: #55708c !important;

    font-size: 18px;

    font-weight: 500;

    margin-bottom: 22px;
}}


/* =======================================================
   HEADINGS
   ======================================================= */

h2,
h3 {{
    color: #174d7a !important;
}}


/* =======================================================
   NORMAL TEXT
   ======================================================= */

p {{
    color: #263b50 !important;
}}


/* =======================================================
   CLINICAL ASSESSMENT BAR
   ======================================================= */

.assessment-bar {{
    background: rgba(255, 255, 255, 0.94);

    border: 1px solid #d7e6f2;

    border-left: 5px solid #1976c9;

    border-radius: 14px;

    padding: 14px 20px;

    margin: 10px 0 25px 0;

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 20px;

    box-shadow:
        0 5px 18px rgba(34, 79, 112, 0.09);

    backdrop-filter: blur(5px);
}}


.assessment-left {{
    display: flex;

    align-items: center;

    gap: 12px;
}}


.assessment-icon {{
    font-size: 28px;

    line-height: 1;
}}


.assessment-title {{
    color: #125ca0;

    font-size: 17px;

    font-weight: 800;
}}


.assessment-subtitle {{
    color: #5a7187;

    font-size: 13px;

    margin-top: 3px;
}}


.assessment-badges {{
    display: flex;

    gap: 8px;

    flex-wrap: wrap;

    justify-content: flex-end;
}}


.assessment-badges span {{
    background: #eef7ff;

    color: #1769aa;

    border: 1px solid #cfe4f6;

    border-radius: 20px;

    padding: 6px 10px;

    font-size: 12px;

    font-weight: 700;

    white-space: nowrap;
}}


/* =======================================================
   PATIENT INFORMATION CARD
   ======================================================= */

.patient-card {{
    background: rgba(255, 255, 255, 0.94);

    border: 1px solid #d7e6f2;

    border-radius: 18px;

    padding: 25px;

    margin-top: 10px;

    box-shadow:
        0 8px 30px rgba(37, 76, 110, 0.12);

    backdrop-filter: blur(6px);
}}


/* =======================================================
   PATIENT HEADING
   ======================================================= */

.patient-heading {{
    color: #125ca0;

    font-size: 27px;

    font-weight: 800;

    margin-bottom: 18px;
}}


/* =======================================================
   INPUT LABELS
   ======================================================= */

[data-testid="stWidgetLabel"] p {{
    color: #173e63 !important;

    font-weight: 700 !important;

    font-size: 15px !important;
}}


/* =======================================================
   NUMBER INPUT
   ======================================================= */

div[data-baseweb="input"] {{
    background-color: #ffffff !important;

    border: 1px solid #c5d8e9 !important;

    border-radius: 10px !important;
}}


div[data-baseweb="input"] > div {{
    background-color: #ffffff !important;
}}


div[data-baseweb="input"] input {{
    background-color: #ffffff !important;

    color: #14283d !important;

    -webkit-text-fill-color: #14283d !important;

    font-size: 16px !important;
}}


/* Number input +/- buttons */

div[data-baseweb="input"] button {{
    background-color: #ffffff !important;

    color: #173e63 !important;
}}


div[data-baseweb="input"] button:hover {{
    background-color: #eaf5ff !important;
}}


/* =======================================================
   SELECT BOX
   ======================================================= */

div[data-baseweb="select"] {{
    background-color: transparent !important;
}}


div[data-baseweb="select"] > div {{
    background-color: #ffffff !important;

    border: 1px solid #c5d8e9 !important;

    border-radius: 10px !important;
}}


div[data-baseweb="select"] span {{
    color: #14283d !important;

    -webkit-text-fill-color: #14283d !important;
}}


div[data-baseweb="select"] input {{
    color: #14283d !important;

    -webkit-text-fill-color: #14283d !important;
}}


/* Dropdown arrow */

div[data-baseweb="select"] svg {{
    fill: #173e63 !important;
}}


/* =======================================================
   DROPDOWN MENU
   ======================================================= */

ul[role="listbox"] {{
    background-color: #ffffff !important;

    border: 1px solid #c5d8e9 !important;

    border-radius: 10px !important;
}}


li[role="option"] {{
    background-color: #ffffff !important;

    color: #14283d !important;

    -webkit-text-fill-color: #14283d !important;
}}


li[role="option"]:hover {{
    background-color: #e8f3ff !important;

    color: #125da0 !important;
}}


/* =======================================================
   PREDICT BUTTON
   ======================================================= */

.stButton {{
    display: flex;

    justify-content: center;

    margin-top: 28px;

    margin-bottom: 15px;
}}


.stButton > button {{
    width: 320px !important;

    height: 55px !important;

    background:
        linear-gradient(
            90deg,
            #1671c4,
            #0b5ca8
        ) !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 12px !important;

    font-size: 18px !important;

    font-weight: 700 !important;

    box-shadow:
        0 6px 18px rgba(0, 91, 160, 0.25);

    transition: all 0.25s ease;
}}


.stButton > button p {{
    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;
}}


.stButton > button span {{
    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;
}}


.stButton > button:hover {{
    background:
        linear-gradient(
            90deg,
            #0b5ca8,
            #07457f
        ) !important;

    transform: translateY(-2px);

    box-shadow:
        0 9px 22px rgba(0, 91, 160, 0.30);
}}


/* =======================================================
   RESULT ALERT
   ======================================================= */

[data-testid="stAlert"] {{
    border-radius: 12px !important;

    margin-top: 20px !important;
}}


[data-testid="stAlert"] p {{
    font-weight: 700 !important;
}}


/* =======================================================
   MEDICAL INFORMATION CARD
   ======================================================= */

.medical-note {{
    background: rgba(255, 255, 255, 0.95);

    border: 1px solid #d6e5f2;

    border-left: 5px solid #1976c9;

    border-radius: 12px;

    padding: 18px 22px;

    margin-top: 22px;

    box-shadow:
        0 5px 18px rgba(34, 79, 112, 0.08);
}}


.medical-note-title {{
    color: #125ca0 !important;

    font-size: 18px;

    font-weight: 800;

    margin-bottom: 8px;
}}


.medical-note-text {{
    color: #52687b !important;

    font-size: 15px;

    line-height: 1.6;
}}


/* =======================================================
   FOOTER
   ======================================================= */

.footer {{
    text-align: center;

    color: #668096;

    font-size: 12px;

    margin-top: 30px;

    padding-top: 10px;
}}


/* =======================================================
   MOBILE
   ======================================================= */

@media (max-width: 768px) {{

    h1 {{
        font-size: 30px !important;
    }}

    .subtitle {{
        font-size: 16px;
    }}

    .assessment-bar {{
        flex-direction: column;

        align-items: flex-start;

        gap: 12px;
    }}

    .assessment-badges {{
        justify-content: flex-start;
    }}

    .patient-card {{
        padding: 15px;
    }}

    .stButton > button {{
        width: 100% !important;
    }}

}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<h1>❤️ Heart Disease Prediction System</h1>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
🩺 Smart health screening powered by machine learning
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# CLINICAL ASSESSMENT BAR
# =========================================================

st.markdown(
    """
<div class="assessment-bar">

    <div class="assessment-left">

        <div class="assessment-icon">
            🩺
        </div>

        <div>

            <div class="assessment-title">
                Clinical Risk Assessment
            </div>

            <div class="assessment-subtitle">
                Enter patient information below to estimate heart disease risk
            </div>

        </div>

    </div>


    <div class="assessment-badges">

        <span>✓ AI-Assisted</span>

        <span>✓ Secure Screening</span>

        <span>⚕️ Not a Diagnosis</span>

    </div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    """
<div class="patient-card">

<div class="patient-heading">
🩺 Patient Information
</div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# INPUTS - TWO COLUMNS
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=40
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    max_hr = st.number_input(
        "Maximum Heart Rate (bpm)",
        min_value=60,
        max_value=250,
        value=150
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )


# =========================================================
# ADDITIONAL INPUTS
# =========================================================

col3, col4, col5 = st.columns(3)


with col3:

    rest_ecg = st.selectbox(
        "Resting ECG",
        ["LVH", "Normal", "ST"]
    )


with col4:

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["No", "Yes"]
    )


with col5:

    st_slope = st.selectbox(
        "ST Slope",
        ["Down", "Flat", "Up"]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("❤️  Predict Heart Disease"):

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    data = {

        "Age": age,

        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,

        "FastingBS": fasting_bs,

        "MaxHR": max_hr,

        "Oldpeak": oldpeak,

        "Sex_M":
            1 if sex == "Male" else 0,

        "ChestPainType_ATA":
            1 if chest_pain == "ATA" else 0,

        "ChestPainType_NAP":
            1 if chest_pain == "NAP" else 0,

        "ChestPainType_TA":
            1 if chest_pain == "TA" else 0,

        "RestingECG_Normal":
            1 if rest_ecg == "Normal" else 0,

        "RestingECG_ST":
            1 if rest_ecg == "ST" else 0,

        "ExerciseAngina_Y":
            1 if exercise_angina == "Yes" else 0,

        "ST_Slope_Flat":
            1 if st_slope == "Flat" else 0,

        "ST_Slope_Up":
            1 if st_slope == "Up" else 0
    }


    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    df = pd.DataFrame([data])


    # -----------------------------------------------------
    # SCALE DATA
    # -----------------------------------------------------

    scaled_data = scaler.transform(df)


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_data)[0]


    # =====================================================
    # HIGH RISK
    # =====================================================

    if prediction == 1:

        st.error(
            "💔 High Risk of Heart Disease ⚠️"
        )

        st.markdown(
            """
<div class="medical-note">

<div class="medical-note-title">
⚠️ Important Screening Result
</div>

<div class="medical-note-text">
The machine-learning model indicates a potentially elevated
risk based on the information provided. This result is for
educational and screening purposes only and is not a medical
diagnosis. Please consult a qualified healthcare professional
for proper evaluation.
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.snow()


    # =====================================================
    # LOW RISK
    # =====================================================

    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

        st.markdown(
            """
<div class="medical-note">

<div class="medical-note-title">
✅ Screening Result
</div>

<div class="medical-note-text">
Based on the information provided, the model predicts a lower
risk of heart disease. This is only a machine-learning
screening result and does not replace professional medical
evaluation.
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.balloons()


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.markdown(
    """
<div class="medical-note">

<div class="medical-note-title">
🩺 Medical Disclaimer
</div>

<div class="medical-note-text">
This application is intended for educational and screening
purposes only. It does not provide a medical diagnosis,
treatment plan, or emergency medical advice.
</div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
HeartCare AI • Machine Learning Based Health Screening
</div>
""",
    unsafe_allow_html=True
)
