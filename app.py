import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

/* ===========================
   BACKGROUND
=========================== */

.stApp{
    background:#edf5f3;
}

.main .block-container{
    padding-top:1rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* ===========================
   HERO SECTION
=========================== */

.hero{
    background:#071d18;
    padding:35px;
    border-radius:20px;
    color:white;
    box-shadow:0px 10px 25px rgba(0,0,0,.25);
    margin-bottom:25px;
}

.hero h1{
    color:white !important;
    font-size:48px;
    font-weight:bold;
}

.hero h5{
    color:#69d9cd !important;
    letter-spacing:4px;
}

.hero p{
    color:#d8f7f4 !important;
    font-size:20px;
}

.ecg{
    color:#40d7c7;
    font-size:26px;
    letter-spacing:3px;
    margin-top:20px;
}

/* ===========================
   CARDS
=========================== */

.section{
    background:white;
    border-radius:18px;
    padding:25px;
    margin-bottom:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,.08);
    color:#1f2937;
}

.result{
    background:white;
    border-radius:18px;
    padding:25px;
    box-shadow:0px 4px 15px rgba(0,0,0,.08);
    color:#1f2937;
}

/* ===========================
   HEADINGS
=========================== */

h1,h2,h3,h4,h5,h6{
    color:#1f2937 !important;
}

p{
    color:#4b5563 !important;
}

/* Hero keeps white text */

.hero h1,
.hero h2,
.hero h3,
.hero h4,
.hero h5,
.hero p{
    color:white !important;
}

.hero h5{
    color:#69d9cd !important;
}

/* ===========================
   LABELS
=========================== */

label{
    color:#1f2937 !important;
    font-weight:600;
}

/* ===========================
   INPUT BOXES
=========================== */

.stNumberInput input{
    background:#242530;
    color:white !important;
    border-radius:10px;
}

.stSelectbox div[data-baseweb="select"]{
    background:#242530;
    border-radius:10px;
}

/* ===========================
   BUTTON
=========================== */

div.stButton>button{

    width:100%;
    height:60px;

    border:none;

    border-radius:12px;

    background:#06382d;

    color:white;

    font-size:22px;

    font-weight:bold;

    transition:.3s;

}

div.stButton>button:hover{

    background:#0b6856;

    color:white;

}

/* ===========================
   METRICS
=========================== */

[data-testid="stMetric"]{

    background:white;

    padding:15px;

    border-radius:15px;

    box-shadow:0px 4px 10px rgba(0,0,0,.08);

}

[data-testid="stMetricLabel"]{

    color:#6b7280 !important;

    font-size:14px;

}

[data-testid="stMetricValue"]{

    color:#111827 !important;

    font-size:34px;

    font-weight:bold;

}

/* ===========================
   SUCCESS / ERROR
=========================== */

.stAlert{

    border-radius:12px;

}

/* ===========================
   TABLE
=========================== */

table{

    border-radius:10px;

    overflow:hidden;

}

/* ===========================
   SIDEBAR
=========================== */

section[data-testid="stSidebar"]{

    background:#071d18;

}

section[data-testid="stSidebar"] *{

    color:white;

}

/* ===========================
   FOOTER
=========================== */

footer{

    visibility:hidden;

}

</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:

    st.title("❤️ Heart AI")

    st.markdown("---")

    st.success("Machine Learning Project")

    st.write("### Model")
    st.info("K-Nearest Neighbors")

    st.write("### Technology")

    st.write("""
- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
""")

    st.markdown("---")

    st.warning(
        "This tool is intended for educational purposes only."
    )

# ==============================
# HERO SECTION
# ==============================

st.markdown("""
<div class="hero">

<h4 style="letter-spacing:4px;color:#A7F3D0;">
AI POWERED CLINICAL SUPPORT
</h4>

<h1>
❤️ Heart Disease Prediction System
</h1>

<p style="font-size:20px;">
Predict the likelihood of heart disease using Machine Learning.
</p>

<hr>

<h3>
🫀 ───────╱╲────────────╱╲────────────╱╲───────
</h3>

</div>
""", unsafe_allow_html=True)

# ==============================
# DASHBOARD METRICS
# ==============================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Patients", "918")

with m2:
    st.metric("Algorithm", "KNN")

with m3:
    st.metric("Accuracy", "87%")

with m4:
    st.metric("Status", "🟢 Online")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================

left, right = st.columns([2, 1])

# -------------------------------
# LEFT COLUMN
# -------------------------------

with left:

    st.markdown("""
    <div class="card">
    <h2>👤 Patient Profile</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=45
        )

    with col2:
        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    col1, col2 = st.columns(2)

    with col1:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "ASY",
                "ATA",
                "NAP",
                "TA"
            ]
        )

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "LVH",
                "ST"
            ]
        )

    st.markdown("---")

    st.markdown("""
    <div class="card">
    <h2>🩺 Vitals & Laboratory</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        resting_bp = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=120
        )

    with col2:

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=700,
            value=200
        )

    col1, col2 = st.columns(2)

    with col1:

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    with col2:

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

    fasting_bs = st.checkbox(
        "Fasting Blood Sugar > 120 mg/dL"
    )

    st.markdown("---")

    st.markdown("""
    <div class="card">
    <h2>📈 Cardiac Test Results</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        st_slope = st.selectbox(
            "ST Slope",
            [
                "Up",
                "Flat",
                "Down"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button("❤️ Run Diagnosis")

# -------------------------------
# RIGHT COLUMN
# -------------------------------

with right:

    st.markdown("""
    <div class="card">

    <h2 align="center">
    ❤️ Prediction Panel
    </h2>

    <br>

    <h1 align="center">
    🫀
    </h1>

    <hr>

    <h4 align="center">

    Fill the patient information

    and press

    <br><br>

    <span style="color:#0F766E;">
    Run Diagnosis
    </span>

    </h4>

    </div>

    """, unsafe_allow_html=True)


# =====================================================
# PREDICTION LOGIC
# =====================================================

if predict:

    with st.spinner("Analyzing patient data..."):

        # -----------------------
        # One-Hot Encoding
        # -----------------------

        sex_m = 1 if sex == "Male" else 0

        chest_ata = 1 if chest_pain == "ATA" else 0
        chest_nap = 1 if chest_pain == "NAP" else 0
        chest_ta = 1 if chest_pain == "TA" else 0

        ecg_normal = 1 if resting_ecg == "Normal" else 0
        ecg_st = 1 if resting_ecg == "ST" else 0

        exercise_y = 1 if exercise_angina == "Yes" else 0

        slope_flat = 1 if st_slope == "Flat" else 0
        slope_up = 1 if st_slope == "Up" else 0

        # -----------------------
        # Model Input
        # -----------------------

        input_data = pd.DataFrame({

            "Age":[age],
            "RestingBP":[resting_bp],
            "Cholesterol":[cholesterol],
            "FastingBS":[int(fasting_bs)],
            "MaxHR":[max_hr],
            "Oldpeak":[oldpeak],

            "Sex_M":[sex_m],

            "ChestPainType_ATA":[chest_ata],
            "ChestPainType_NAP":[chest_nap],
            "ChestPainType_TA":[chest_ta],

            "RestingECG_Normal":[ecg_normal],
            "RestingECG_ST":[ecg_st],

            "ExerciseAngina_Y":[exercise_y],

            "ST_Slope_Flat":[slope_flat],
            "ST_Slope_Up":[slope_up]

        })

        # -----------------------
        # Scaling
        # -----------------------

        scaled_data = scaler.transform(input_data)

        # -----------------------
        # Prediction
        # -----------------------

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(scaled_data)[0][1]

        risk_score = probability * 100
        health_score = (1 - probability) * 100

# =====================================================
# RESULT DASHBOARD
# =====================================================

with right:

    st.markdown("## ❤️ Prediction Result")

    if predict:

        st.subheader("📊 Risk Probability")
        st.progress(float(probability))
        st.write(f"### {risk_score:.1f}%")

        if probability < 0.30:
            st.success("🟢 LOW RISK")
        elif probability < 0.70:
            st.warning("🟡 MODERATE RISK")
        else:
            st.error("🔴 HIGH RISK")

        st.markdown("---")

        if prediction == 1:

            st.error("🚨 High Risk of Heart Disease")

            st.metric("Risk Score", f"{risk_score:.1f}%")

            st.warning(
                "Please consult a Cardiologist for further medical evaluation."
            )

            st.toast("🚨 High Risk Detected")
            st.snow()

            st.error("""
### 🚑 Recommendation

- Visit a Cardiologist
- ECG Test
- Blood Test
- Follow Doctor's Advice
- Maintain a Healthy Lifestyle
""")

        else:

            st.success("🎉 Low Risk of Heart Disease")

            st.metric(
                "Heart Health Score",
                f"{health_score:.1f}%"
            )

            st.success(
                "Your prediction indicates a low risk of heart disease."
            )

            st.balloons()
            st.toast("❤️ Low Risk")

            st.success("""
### ❤️ Recommendation

- Eat a Healthy Diet
- Exercise Regularly
- Sleep 7–8 Hours
- Avoid Smoking
- Annual Health Checkup
""")

    else:
        st.info("👈 Enter patient details and click **Run Diagnosis**.")


# =====================================================
# PATIENT SUMMARY
# =====================================================

if predict:

    st.markdown("---")

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Age",

            "Gender",

            "Blood Pressure",

            "Cholesterol",

            "Maximum Heart Rate",

            "OldPeak",

            "Exercise Angina"

        ],

        "Value":[

            age,

            sex,

            resting_bp,

            cholesterol,

            max_hr,

            oldpeak,

            exercise_angina

        ]

    })

    st.table(summary)

# =====================================================
# MODEL INFORMATION
# =====================================================

st.markdown("---")


# =====================================================
# DOWNLOAD REPORT
# =====================================================

if predict:

    report = f"""
=========================================
 HEART DISEASE PREDICTION REPORT
=========================================

Patient Details
-------------------------
Age                 : {age}
Gender              : {sex}

Resting BP          : {resting_bp}
Cholesterol         : {cholesterol}

Maximum Heart Rate  : {max_hr}
OldPeak             : {oldpeak}

Fasting Blood Sugar : {"Yes" if fasting_bs else "No"}

Chest Pain Type     : {chest_pain}
Resting ECG         : {resting_ecg}
Exercise Angina     : {exercise_angina}
ST Slope            : {st_slope}

=========================================

Prediction

{"HIGH RISK OF HEART DISEASE" if prediction==1 else "LOW RISK OF HEART DISEASE"}

Risk Probability

{risk_score:.2f} %

Heart Health Score

{health_score:.2f} %

=========================================

This prediction is generated using a
Machine Learning KNN model.

This application is intended only for
educational purposes and should not be
used as a medical diagnosis.

=========================================
"""

    st.download_button(
        label="📄 Download Prediction Report",
        data=report,
        file_name="Heart_Disease_Report.pdf",
        mime="text/plain"
    )

# =====================================================
# CURRENT DATE & TIME
# =====================================================

from datetime import datetime

st.markdown("---")

st.write(
    "🕒 Prediction Time :",
    datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
)

# =====================================================
# DISCLAIMER
# =====================================================

st.warning("""
### ⚠ Medical Disclaimer

This application is developed for educational purposes only.

The prediction generated by this model should NOT be considered
a medical diagnosis.

Always consult a qualified healthcare professional for proper
medical advice and treatment.
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

<h3>❤️ Heart Disease Prediction System</h3>

Developed using

<b>Python • Streamlit • Scikit-Learn • Pandas • NumPy</b>

<br><br>

© 2026 Ankit Rai

</div>
""",
unsafe_allow_html=True
)
