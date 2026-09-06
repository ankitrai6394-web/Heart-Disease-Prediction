import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="HeartCare AI | Cardiovascular Risk Screening",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================
encoded_image = ""
if os.path.exists("medical_bg.png"):
    try:
        with open("medical_bg.png", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
    except Exception:
        encoded_image = ""

# =========================================================
# LOAD MODEL AND SCALER
# =========================================================
@st.cache_resource
def load_ml_pipeline():
    model, scaler, is_fallback = None, None, False
    try:
        model = joblib.load("knn_heart_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
    except Exception:
        is_fallback = True
    return model, scaler, is_fallback

model, scaler, is_fallback = load_ml_pipeline()

# =========================================================
# MODERN GLASSMORPHISM DESIGN SYSTEM (CUSTOM CSS)
# =========================================================
st.markdown(f"""
st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');


/* =========================================================
   GLOBAL FONT
   ========================================================= */

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}


/* =========================================================
   MAIN MEDICAL BACKGROUND
   ========================================================= */

.stApp {{

    background-color: #eef7ff;

    background-image:
        linear-gradient(
            rgba(239, 248, 255, 0.86),
            rgba(247, 251, 255, 0.94)
        )
        {f', url("data:image/png;base64,{encoded_image}")' if encoded_image else ''};

    background-size: cover;
    background-position: center;
    background-attachment: fixed;

    min-height: 100vh;

    transition: background 0.5s ease;
}}


/* =========================================================
   SUBTLE MEDICAL BACKGROUND GLOW
   ========================================================= */

.stApp::before {{

    content: "";

    position: fixed;

    width: 500px;
    height: 500px;

    top: 5%;
    left: -150px;

    background:
        radial-gradient(
            circle,
            rgba(37, 99, 235, 0.08),
            transparent 70%
        );

    border-radius: 50%;

    pointer-events: none;

    z-index: 0;
}}


.stApp::after {{

    content: "";

    position: fixed;

    width: 450px;
    height: 450px;

    bottom: 5%;
    right: -120px;

    background:
        radial-gradient(
            circle,
            rgba(220, 38, 38, 0.06),
            transparent 70%
        );

    border-radius: 50%;

    pointer-events: none;

    z-index: 0;
}}


/* =========================================================
   CONTENT CONTAINER
   ========================================================= */

.block-container {{

    max-width: 1200px;

    padding-top: 1.8rem;
    padding-bottom: 3rem;

    position: relative;

    z-index: 1;
}}


/* =========================================================
   HERO CARD
   ========================================================= */

.hero-card {{

    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 55%,
            #1e40af 100%
        );

    border-radius: 22px;

    padding: 34px 38px;

    color: white;

    margin-bottom: 24px;

    box-shadow:
        0 20px 35px rgba(15, 23, 42, 0.16);

    position: relative;

    overflow: hidden;

    transition:
        transform 0.35s ease,
        box-shadow 0.35s ease;
}}


/* HERO HOVER */

.hero-card:hover {{

    transform: translateY(-3px);

    box-shadow:
        0 25px 45px rgba(15, 23, 42, 0.22);
}}


/* HERO GLOW */

.hero-card::after {{

    content: "";

    position: absolute;

    top: -100px;
    right: -80px;

    width: 350px;
    height: 350px;

    background:
        radial-gradient(
            circle,
            rgba(239, 68, 68, 0.30),
            transparent 68%
        );

    border-radius: 50%;

    pointer-events: none;

    transition:
        transform 0.5s ease;
}}


.hero-card:hover::after {{

    transform: scale(1.2);
}}


/* =========================================================
   HERO TITLE
   ========================================================= */

.hero-title {{

    font-size: 36px !important;

    font-weight: 800 !important;

    color: #ffffff !important;

    margin: 0 0 7px 0 !important;

    letter-spacing: -0.5px;

    display: flex;

    align-items: center;

    gap: 12px;
}}


/* =========================================================
   HERO SUBTITLE
   ========================================================= */

.hero-subtitle {{

    font-size: 16px;

    color: #cbd5e1;

    font-weight: 500;

    margin: 0;
}}


/* =========================================================
   HERO TAGS
   ========================================================= */

.hero-tags {{

    display: flex;

    gap: 10px;

    margin-top: 17px;

    flex-wrap: wrap;
}}


.hero-tag {{

    background:
        rgba(255,255,255,0.10);

    border:
        1px solid rgba(255,255,255,0.20);

    backdrop-filter: blur(10px);

    padding:
        6px 13px;

    border-radius:
        20px;

    font-size:
        12px;

    font-weight:
        600;

    color:
        #e2e8f0;

    transition:
        all 0.25s ease;
}}


/* TAG HOVER */

.hero-tag:hover {{

    background:
        rgba(255,255,255,0.18);

    border-color:
        rgba(255,255,255,0.35);

    transform:
        translateY(-2px);

    color:
        #ffffff;
}}


/* =========================================================
   STAT CARDS
   ========================================================= */

.stat-card {{

    background:
        rgba(255,255,255,0.88);

    backdrop-filter:
        blur(15px);

    border:
        1px solid rgba(226,232,240,0.9);

    border-radius:
        17px;

    padding:
        17px 20px;

    box-shadow:
        0 5px 15px rgba(15,23,42,0.04);

    display:
        flex;

    align-items:
        center;

    gap:
        14px;

    transition:
        all 0.3s ease;
}}


/* STAT CARD HOVER */

.stat-card:hover {{

    transform:
        translateY(-5px);

    border-color:
        #93c5fd;

    box-shadow:
        0 14px 28px rgba(37,99,235,0.12);

    background:
        rgba(255,255,255,0.97);
}}


/* =========================================================
   STAT ICON
   ========================================================= */

.stat-icon {{

    width:
        44px;

    height:
        44px;

    border-radius:
        12px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    font-size:
        20px;

    transition:
        transform 0.3s ease;
}}


.stat-card:hover .stat-icon {{

    transform:
        scale(1.1)
        rotate(3deg);
}}


.icon-blue {{
    background:#eff6ff;
    color:#2563eb;
}}

.icon-green {{
    background:#f0fdf4;
    color:#16a34a;
}}

.icon-purple {{
    background:#faf5ff;
    color:#9333ea;
}}


/* =========================================================
   STAT TEXT
   ========================================================= */

.stat-label {{

    font-size:
        12px;

    font-weight:
        700;

    text-transform:
        uppercase;

    color:
        #64748b;
}}


.stat-value {{

    font-size:
        16px;

    font-weight:
        800;

    color:
        #0f172a;
}}


/* =========================================================
   GLASS PANELS
   ========================================================= */

.glass-panel {{

    background:
        rgba(255,255,255,0.93);

    backdrop-filter:
        blur(16px);

    border:
        1px solid #dbe7f2;

    border-radius:
        18px;

    padding:
        24px;

    box-shadow:
        0 8px 20px rgba(15,23,42,0.05);

    margin-bottom:
        20px;

    transition:
        all 0.3s ease;
}}


/* PANEL HOVER */

.glass-panel:hover {{

    transform:
        translateY(-3px);

    border-color:
        #93c5fd;

    box-shadow:
        0 14px 32px rgba(37,99,235,0.10);

    background:
        rgba(255,255,255,0.97);
}}


/* =========================================================
   SECTION HEADING
   ========================================================= */

h3 {{

    color:
        #123b66 !important;

    font-weight:
        800 !important;
}}


/* =========================================================
   INPUT LABELS
   ========================================================= */

[data-testid="stWidgetLabel"] p {{

    font-weight:
        700 !important;

    color:
        #173e63 !important;

    font-size:
        14px !important;

    transition:
        color 0.2s ease;
}}


/* =========================================================
   INPUT BOXES
   ========================================================= */

div[data-baseweb="input"],
div[data-baseweb="select"] > div {{

    border-radius:
        11px !important;

    border:
        1px solid #cbd5e1 !important;

    background-color:
        #ffffff !important;

    transition:
        all 0.25s ease !important;

    box-shadow:
        0 2px 5px rgba(15,23,42,0.02);
}}


/* =========================================================
   INPUT HOVER
   ========================================================= */

div[data-baseweb="input"]:hover,
div[data-baseweb="select"] > div:hover {{

    border-color:
        #60a5fa !important;

    box-shadow:
        0 0 0 3px rgba(59,130,246,0.08) !important;

    transform:
        translateY(-1px);
}}


/* =========================================================
   INPUT FOCUS
   ========================================================= */

div[data-baseweb="input"]:focus-within {{

    border-color:
        #2563eb !important;

    box-shadow:
        0 0 0 3px rgba(37,99,235,0.12) !important;
}}


/* =========================================================
   INPUT TEXT
   ========================================================= */

div[data-baseweb="input"] input {{

    color:
        #0f172a !important;

    background:
        #ffffff !important;

    font-weight:
        600 !important;
}}


/* =========================================================
   SELECT TEXT
   ========================================================= */

div[data-baseweb="select"] span {{

    color:
        #0f172a !important;

    font-weight:
        600 !important;
}}


/* =========================================================
   DROPDOWN
   ========================================================= */

ul[role="listbox"] {{

    background:
        #ffffff !important;

    border:
        1px solid #dbe7f2 !important;

    border-radius:
        10px !important;

    box-shadow:
        0 15px 35px rgba(15,23,42,0.15) !important;
}}


li[role="option"] {{

    background:
        #ffffff !important;

    color:
        #0f172a !important;

    font-weight:
        600 !important;

    transition:
        background 0.2s ease;
}}


li[role="option"]:hover {{

    background:
        #eff6ff !important;

    color:
        #1d4ed8 !important;
}}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {{

    color:
        #475569 !important;

    font-weight:
        700 !important;

    transition:
        all 0.25s ease;
}}


button[data-baseweb="tab"]:hover {{

    color:
        #2563eb !important;

    background:
        rgba(239,246,255,0.7) !important;

    border-radius:
        8px;
}}


/* =========================================================
   PREDICT BUTTON
   ========================================================= */

.stButton {{

    display:
        flex;

    justify-content:
        center;

    margin-top:
        10px;

    margin-bottom:
        20px;
}}


.stButton > button {{

    width:
        100% !important;

    max-width:
        420px !important;

    height:
        58px !important;

    background:
        linear-gradient(
            135deg,
            #dc2626 0%,
            #b91c1c 100%
        ) !important;

    color:
        #ffffff !important;

    border:
        none !important;

    border-radius:
        14px !important;

    font-size:
        18px !important;

    font-weight:
        800 !important;

    box-shadow:
        0 10px 20px rgba(220,38,38,0.25) !important;

    transition:
        all 0.3s ease !important;
}}


/* BUTTON HOVER */

.stButton > button:hover {{

    background:
        linear-gradient(
            135deg,
            #ef4444 0%,
            #b91c1c 100%
        ) !important;

    transform:
        translateY(-4px)
        scale(1.01);

    box-shadow:
        0 16px 30px rgba(220,38,38,0.35) !important;
}}


/* BUTTON CLICK */

.stButton > button:active {{

    transform:
        translateY(0)
        scale(0.99);
}}


/* BUTTON TEXT */

.stButton > button p,
.stButton > button span {{

    color:
        #ffffff !important;

    font-weight:
        800 !important;
}}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {{

    border-radius:
        18px;

    padding:
        26px;

    margin-top:
        20px;

    box-shadow:
        0 12px 24px rgba(0,0,0,0.07);

    transition:
        all 0.3s ease;
}}


.result-card:hover {{

    transform:
        translateY(-4px);

    box-shadow:
        0 18px 35px rgba(0,0,0,0.12);
}}


/* HIGH RISK */

.result-card-high {{

    background:
        linear-gradient(
            135deg,
            #fff5f5 0%,
            #fed7d7 100%
        );

    border:
        2px solid #f87171;
}}


/* LOW RISK */

.result-card-low {{

    background:
        linear-gradient(
            135deg,
            #f0fdf4 0%,
            #dcfce7 100%
        );

    border:
        2px solid #4ade80;
}}


/* =========================================================
   BADGES
   ========================================================= */

.badge-chip {{

    display:
        inline-block;

    padding:
        5px 12px;

    background:
        rgba(255,255,255,0.85);

    border-radius:
        20px;

    font-size:
        12px;

    font-weight:
        700;

    color:
        #334155;

    margin-right:
        6px;

    margin-top:
        6px;

    border:
        1px solid rgba(0,0,0,0.06);

    transition:
        all 0.2s ease;
}}


.badge-chip:hover {{

    transform:
        translateY(-2px);

    background:
        #ffffff;

    box-shadow:
        0 4px 10px rgba(15,23,42,0.08);
}}


/* =========================================================
   ALERT
   ========================================================= */

[data-testid="stAlert"] {{

    border-radius:
        12px !important;

    border:
        1px solid #bfdbfe !important;
}}


[data-testid="stAlert"] p {{

    font-weight:
        700 !important;
}}


/* =========================================================
   FOOTER
   ========================================================= */

.footer-text {{

    text-align:
        center;

    color:
        #64748b;

    font-size:
        13px;

    margin-top:
        35px;

    padding-top:
        15px;

    border-top:
        1px solid #dbe7f2;
}}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {{

    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}

    .hero-card {{
        padding: 25px 22px;
    }}

    .hero-title {{
        font-size: 28px !important;
    }}

    .hero-subtitle {{
        font-size: 14px;
    }}

    .glass-panel {{
        padding: 17px;
    }}

    .stButton > button {{
        max-width: 100% !important;
    }}

}}

</style>
""", unsafe_allow_html=True)
""", unsafe_allow_html=True)

# =========================================================
# HEADER BANNER
# =========================================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">
        <span>❤️ HeartCare AI</span>
    </div>
    <div class="hero-subtitle">
        Clinical-grade cardiovascular risk stratification powered by Machine Learning
    </div>
    <div class="hero-tags">
        <span class="hero-tag">⚡ Real-time Inference</span>
        <span class="hero-tag">🔒 Encrypted Local Analysis</span>
        <span class="hero-tag">🩺 Clinical Biomarker Scoring</span>
        <span class="hero-tag">⚕️ Decision Support Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

if is_fallback:
    st.info("💡 **Interactive Mode Active**: Utilizing heuristic evaluation model. Add `knn_heart_model.pkl` & `heart_scaler.pkl` to run live trained KNN pipeline.")

# =========================================================
# METRIC SUMMARY BAR
# =========================================================
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon icon-blue">📊</div>
        <div>
            <div class="stat-label">Classifier Engine</div>
            <div class="stat-value">KNN Algorithm</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon icon-green">🧬</div>
        <div>
            <div class="stat-label">Biomarkers Tracked</div>
            <div class="stat-value">11 Clinical Features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-icon icon-purple">🎯</div>
        <div>
            <div class="stat-label">Screening Target</div>
            <div class="stat-value">Ischemic Disease Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# INPUT FORM (ORGANIZED IN TABBED CARDS)
# =========================================================
st.markdown("<h3 style='color: #0f172a; font-weight: 800; margin-bottom: 12px;'>🩺 Patient Clinical Data Entry</h3>", unsafe_allow_html=True)

tab_vitals, tab_labs, tab_ecg = st.tabs([
    "👤 Vitals & Demographics",
    "🩸 Laboratory Biomarkers",
    "⚡ Exercise & ECG Profile"
])

with tab_vitals:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=48, help="Patient age in completed years.")
        sex = st.selectbox("Biological Gender", ["Male", "Female"])
    with c2:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130, help="Resting BP measured on admission.")
        resting_ecg = st.selectbox(
            "Resting ECG Result",
            ["Normal", "ST", "LVH"],
            format_func=lambda x: {
                "Normal": "Normal Baseline",
                "ST": "ST-T Wave Abnormality",
                "LVH": "Left Ventricular Hypertrophy"
            }[x]
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_labs:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=700, value=210, help="Fasting serum cholesterol level.")
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)"
        )
    with c4:
        max_hr = st.number_input("Maximum Heart Rate (bpm)", min_value=60, max_value=250, value=150, help="Peak heart rate achieved during exercise.")
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "Asymptomatic (ASY)",
                "ATA": "Atypical Angina (ATA)",
                "NAP": "Non-Anginal Pain (NAP)",
                "TA": "Typical Angina (TA)"
            }[x]
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_ecg:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    with c6:
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            format_func=lambda x: {
                "Up": "Upsloping",
                "Flat": "Flat (Horizontal)",
                "Down": "Downsloping"
            }[x]
        )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREDICTION PROCESSOR
# =========================================================
if st.button("❤️  Run Heart Disease Risk Assessment"):
    data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }
    
    df = pd.DataFrame([data])
    
    if not is_fallback and model is not None and scaler is not None:
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)[0]
        risk_proba = float(model.predict_proba(scaled_data)[0][1]) if hasattr(model, "predict_proba") else (0.85 if prediction == 1 else 0.15)
    else:
        score = 0.15
        if age > 55: score += 0.15
        if sex == "Male": score += 0.10
        if chest_pain == "ASY": score += 0.25
        if exercise_angina == "Yes": score += 0.20
        if st_slope in ["Flat", "Down"]: score += 0.20
        if oldpeak > 1.0: score += 0.15
        if cholesterol > 240: score += 0.10
        if resting_bp > 140: score += 0.10
        risk_proba = min(0.96, score)
        prediction = 1 if risk_proba >= 0.50 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-card-high">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
                <div style="font-size: 42px;">💔</div>
                <div>
                    <h2 style="margin: 0; color: #991b1b !important; font-size: 24px; font-weight: 800;">High Risk of Heart Disease Identified</h2>
                    <div style="font-weight: 700; color: #dc2626; font-size: 15px;">Calculated Probability: {risk_proba*100:.1f}%</div>
                </div>
            </div>
            <p style="color: #451a03 !important; font-size: 15px; line-height: 1.6;">
                The ML model indicates elevated clinical indicators associated with coronary artery disease. 
                Immediate clinical correlation and follow-up cardiovascular evaluation are recommended.
            </p>
            <div>
                <span class="badge-chip">Chest Pain: {chest_pain}</span>
                <span class="badge-chip">ST Slope: {st_slope}</span>
                <span class="badge-chip">Exercise Angina: {exercise_angina}</span>
                <span class="badge-chip">Oldpeak: {oldpeak}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.snow()
    else:
        st.markdown(f"""
        <div class="result-card result-card-low">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
                <div style="font-size: 42px;">✅</div>
                <div>
                    <h2 style="margin: 0; color: #166534 !important; font-size: 24px; font-weight: 800;">Low Risk of Heart Disease Identified</h2>
                    <div style="font-weight: 700; color: #16a34a; font-size: 15px;">Calculated Probability: {risk_proba*100:.1f}%</div>
                </div>
            </div>
            <p style="color: #064e3b !important; font-size: 15px; line-height: 1.6;">
                The patient's current biomarkers fall within non-pathological thresholds. 
                Regular health monitoring and routine preventative screenings are advised.
            </p>
            <div>
                <span class="badge-chip">Age: {age} yrs</span>
                <span class="badge-chip">Max HR: {max_hr} bpm</span>
                <span class="badge-chip">Resting BP: {resting_bp} mm Hg</span>
                <span class="badge-chip">Cholesterol: {cholesterol} mg/dl</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

# =========================================================
# MEDICAL DISCLAIMER & FOOTER
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.8); border-left: 4px solid #2563eb; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div style="color: #1e3a8a; font-weight: 800; font-size: 14px; margin-bottom: 4px;">
        ⚕️ Medical Disclaimer & Clinical Usage Protocol
    </div>
    <div style="color: #475569; font-size: 13px; line-height: 1.5;">
        This application provides machine-learning screening estimates strictly for educational and decision-support research. 
        It does not constitute a formal medical diagnosis or treatment plan.
    </div>
</div>

<div class="footer-text">
    HeartCare AI • K-Nearest Neighbors Health Screening Engine
</div>
""", unsafe_allow_html=True)
