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
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
    <style>

        /* =================================================
           PAGE BACKGROUND
           ================================================= */

        .stApp {{
            background:
                linear-gradient(
                    rgba(235, 246, 255, 0.88),
                    rgba(248, 252, 255, 0.94)
                )
                url("data:image/png;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}


        /* =================================================
           MAIN CONTAINER
           ================================================= */

        .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}


        /* =================================================
           TITLE
           ================================================= */

        h1 {{
            color: #123b66 !important;
            text-align: center;
            font-size: 42px !important;
            font-weight: 800 !important;
            margin-bottom: 5px !important;
        }}


        .subtitle {{
            text-align: center;
            color: #55708c !important;
            font-size: 18px;
            margin-bottom: 30px;
        }}


        /* =================================================
           HEADINGS
           ================================================= */

        h2 {{
            color: #174d7a !important;
        }}

        h3 {{
            color: #174d7a !important;
        }}


        /* =================================================
           NORMAL TEXT
           ================================================= */

        p {{
            color: #263b50 !important;
        }}


        /* =================================================
           INPUT LABELS
           ================================================= */

        [data-testid="stWidgetLabel"] p {{
            color: #183b5b !important;
            font-weight: 700 !important;
            font-size: 15px !important;
        }}


        /* =================================================
           NUMBER INPUT
           ================================================= */

        div[data-baseweb="input"] {{
            background-color: rgba(255, 255, 255, 0.97) !important;
            border: 1px solid #c9d9e8 !important;
            border-radius: 10px !important;
        }}

        div[data-baseweb="input"] input {{
            color: #162b40 !important;
            background-color: transparent !important;
            font-size: 16px !important;
        }}


        /* =================================================
           SELECTBOX
           ================================================= */

        div[data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.97) !important;
            border: 1px solid #c9d9e8 !important;
            border-radius: 10px !important;
        }}

        div[data-baseweb="select"] span {{
            color: #162b40 !important;
        }}


        /* =================================================
           DROPDOWN MENU
           ================================================= */

        ul[role="listbox"] {{
            background-color: #ffffff !important;
        }}

        li[role="option"] {{
            color: #162b40 !important;
            background-color: #ffffff !important;
        }}

        li[role="option"]:hover {{
            background-color: #e8f3ff !important;
            color: #125da0 !important;
        }}


        /* =================================================
           PREDICT BUTTON
           ================================================= */

        .stButton {{
            display: flex;
            justify-content: center;
            margin-top: 25px;
        }}

        .stButton > button {{
            width: 320px !important;
            height: 55px !important;

            background: linear-gradient(
                90deg,
                #1671c4,
                #0b5ca8
            ) !important;

            color: white !important;
            border: none !important;
            border-radius: 12px !important;

            font-size: 18px !important;
            font-weight: 700 !important;

            box-shadow:
                0 6px 18px rgba(0, 91, 160, 0.22);

            transition: all 0.25s ease;
        }}

        .stButton > button:hover {{
            background: linear-gradient(
                90deg,
                #0b5ca8,
                #07457f
            ) !important;

            color: white !important;

            transform: translateY(-2px);

            box-shadow:
                0 9px 22px rgba(0, 91, 160, 0.30);
        }}


        /* =================================================
           INPUT CARD
           ================================================= */

        .input-card {{
            background: rgba(255, 255, 255, 0.90);
            border-radius: 18px;

            padding: 25px;

            margin-top: 20px;

            box-shadow:
                0 8px 30px rgba(37, 76, 110, 0.10);

            border:
                1px solid rgba(208, 225, 239, 0.85);
        }}


        /* =================================================
           MEDICAL NOTE
           ================================================= */

        .medical-note {{
            background: rgba(255, 255, 255, 0.94);

            border:
                1px solid #d6e5f2;

            border-left:
                5px solid #1976c9;

            border-radius: 12px;

            padding: 18px 22px;

            margin-top: 30px;

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


        /* =================================================
           ALERT MESSAGES
           ================================================= */

        [data-testid="stAlert"] {{
            border-radius: 12px !important;
        }}

        [data-testid="stAlert"] p {{
            font-weight: 700 !important;
        }}


        /* =================================================
           MOBILE RESPONSIVE
           ================================================= */

        @media (max-width: 768px) {{

            h1 {{
                font-size: 30px !important;
            }}

            .subtitle {{
                font-size: 16px;
            }}

            .stButton > button {{
                width: 100% !important;
            }}

            .input-card {{
                padding: 15px;
            }}
        }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<h1>❤️ Heart Disease Prediction System</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '🩺 Smart health screening powered by machine learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION CARD
# =========================================================

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.markdown("### 🩺 Patient Information")


# =========================================================
# FIRST TWO COLUMNS
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
        value=1.0
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


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("❤️  Predict Heart Disease"):

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


    # =====================================================
    # DATAFRAME
    # =====================================================

    df = pd.DataFrame([data])


    # =====================================================
    # SCALE DATA
    # =====================================================

    scaled_data = scaler.transform(df)


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

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
                    ⚠️ Important Result
                </div>

                <div class="medical-note-text">
                    The machine-learning model indicates a potentially
                    elevated risk based on the information entered.
                    This result is for educational and screening purposes
                    only and is not a medical diagnosis.
                    Please consult a qualified healthcare professional.
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
                    Based on the information provided, the model predicts
                    a lower risk of heart disease. This result is only a
                    machine-learning screening result and does not replace
                    professional medical evaluation.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()


# =========================================================
# FOOTER / DISCLAIMER
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
