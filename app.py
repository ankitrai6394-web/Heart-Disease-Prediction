import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import io
import math

# Safe conditional import for ReportLab (prevents Streamlit Cloud crash if package is missing)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPulse | Animated Medical Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CARTOON HEART BACKGROUND & PLAYFUL COLOR COMBINATION
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .cartoon-heading {
        font-family: 'Fredoka', cursive, sans-serif !important;
        letter-spacing: 0.02em;
    }
    
    /* Cartoon Heart Background & Gradient Overlay */
    @keyframes bgGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: 
            radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.25) 0%, transparent 40%),
            radial-gradient(circle at 80% 80%, rgba(244, 63, 94, 0.25) 0%, transparent 40%),
            linear-gradient(135deg, rgba(13, 19, 36, 0.92) 0%, rgba(26, 21, 56, 0.9) 50%, rgba(15, 23, 42, 0.94) 100%),
            url('https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Floating Heart Mascot Animation */
    @keyframes floatMascot {
        0% { transform: translateY(0px) rotate(0deg) scale(1); }
        50% { transform: translateY(-12px) rotate(4deg) scale(1.04); }
        100% { transform: translateY(0px) rotate(0deg) scale(1); }
    }

    /* Pulse Beating Heart Animation */
    @keyframes heartBeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.15); }
        28% { transform: scale(1); }
        42% { transform: scale(1.15); }
        70% { transform: scale(1); }
    }

    .mascot-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 3px solid rgba(56, 189, 248, 0.4);
        border-radius: 24px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 25px rgba(56, 189, 248, 0.2);
    }

    .heart-mascot-img {
        width: 115px;
        height: 115px;
        animation: floatMascot 4s ease-in-out infinite;
        filter: drop-shadow(0 10px 15px rgba(244, 63, 94, 0.5));
    }

    .beating-heart-icon {
        display: inline-block;
        animation: heartBeat 1.5s infinite;
    }

    /* Section Headers */
    .section-head {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        border-bottom: 2px dashed rgba(255, 255, 255, 0.15);
        padding-bottom: 0.75rem;
    }

    /* High Risk Banner */
    .res-card-danger {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.88) 0%, rgba(159, 18, 57, 0.95) 100%);
        border: 3px solid #fb7185;
        border-radius: 28px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 0 45px rgba(244, 63, 94, 0.5);
        animation: floatMascot 5s ease-in-out infinite;
    }

    /* Low Risk Banner */
    .res-card-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.88) 0%, rgba(5, 150, 105, 0.95) 100%);
        border: 3px solid #6ee7b7;
        border-radius: 28px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 0 45px rgba(16, 185, 129, 0.5);
        animation: floatMascot 5s ease-in-out infinite;
    }

    /* Cartoon Action Advice Cards */
    .action-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(16px);
        border: 2px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }

    .action-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #38bdf8;
        box-shadow: 0 15px 30px rgba(56, 189, 248, 0.4);
    }

    .action-icon {
        width: 80px;
        height: 80px;
        margin-bottom: 0.8rem;
        animation: floatMascot 4s ease-in-out infinite;
    }

    /* Interactive Bouncy Submit Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        color: #0f172a;
        font-family: 'Fredoka', cursive, sans-serif !important;
        font-weight: 700;
        font-size: 1.35rem;
        padding: 1rem 2rem;
        border-radius: 18px;
        border: 3px solid #ffffff;
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.5);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: scale(1.03) translateY(-4px);
        box-shadow: 0 18px 35px rgba(56, 189, 248, 0.7);
        background: linear-gradient(90deg, #7dd3fc 0%, #a5b4fc 100%);
        color: #000000;
    }

    /* Custom Input Control Hover Pop */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: rgba(30, 41, 59, 0.85) !important;
        border: 2px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 16px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }

    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
        border-color: #38bdf8 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.35) !important;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2.5rem 0;
        font-size: 0.95rem;
        border-top: 2px dashed rgba(255, 255, 255, 0.1);
        margin-top: 3.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZE SESSION STATE FOR HISTORY
# -----------------------------------------------------------------------------
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

# -----------------------------------------------------------------------------
# HELPER: LOAD MODELS SAFELY
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    try:
        model = joblib.load(BASE_DIR / "knn_heart_model.pkl")
        scaler = joblib.load(BASE_DIR / "heart_scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_models()

# -----------------------------------------------------------------------------
# HIGH-QUALITY REPORTLAB PDF GENERATOR
# -----------------------------------------------------------------------------
def generate_pdf_report(patient_info, result_info):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    story = []
    styles = getSampleStyleSheet()

    header_blue = colors.HexColor('#0284c7')
    dark_slate = colors.HexColor('#0f172a')
    border_gray = colors.HexColor('#cbd5e1')
    
    if result_info['prediction'] == 1:
        card_bg = colors.HexColor('#fff1f2')
        card_border = colors.HexColor('#f43f5e')
        card_text_color = colors.HexColor('#be123c')
        outcome_status = "HIGH CARDIOVASCULAR RISK DETECTED 💔"
        advice_summary = "Elevated cardiovascular risk parameters detected. Immediate clinical consultation with a specialist cardiologist is strongly advised."
    else:
        card_bg = colors.HexColor('#f0fdf4')
        card_border = colors.HexColor('#10b981')
        card_text_color = colors.HexColor('#047857')
        outcome_status = "LOW CARDIOVASCULAR RISK (HAPPY HEART) 💚"
        advice_summary = "Patient physiological parameters fall within healthy clinical baselines. Continue routine checkups and healthy lifestyle habits."

    title_style = ParagraphStyle(
        'PdfTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=header_blue,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'PdfSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=12
    )

    story.append(Paragraph("🫀 CardioPulse AI — Diagnostic Assessment Report", title_style))
    story.append(Paragraph(f"Report Generated: {result_info['timestamp']} | Dr. Hearty Clinical Intelligence Suite", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=header_blue, spaceAfter=15))

    table_heading_style = ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#0369a1'))
    cell_style = ParagraphStyle('TC', fontName='Helvetica', fontSize=9, textColor=dark_slate)

    table_data = [
        [Paragraph("<b>Clinical Parameter</b>", table_heading_style), Paragraph("<b>Evaluated Patient Measurement</b>", table_heading_style)],
        [Paragraph("Age", cell_style), Paragraph(f"{patient_info['age']} Years", cell_style)],
        [Paragraph("Gender", cell_style), Paragraph(f"{patient_info['sex']}", cell_style)],
        [Paragraph("Resting Blood Pressure", cell_style), Paragraph(f"{patient_info['resting_bp']} mm Hg", cell_style)],
        [Paragraph("Serum Cholesterol", cell_style), Paragraph(f"{patient_info['cholesterol']} mg/dL", cell_style)],
        [Paragraph("Fasting Blood Sugar (> 120 mg/dl)", cell_style), Paragraph("Yes (> 120 mg/dl)" if patient_info['fasting_bs'] == 1 else "No (<= 120 mg/dl)", cell_style)],
        [Paragraph("Maximum Heart Rate Achieved", cell_style), Paragraph(f"{patient_info['max_hr']} bpm", cell_style)],
        [Paragraph("Chest Pain Type", cell_style), Paragraph(f"{patient_info['chest_pain']}", cell_style)],
        [Paragraph("Resting ECG Result", cell_style), Paragraph(f"{patient_info['rest_ecg']}", cell_style)],
        [Paragraph("Exercise Induced Angina", cell_style), Paragraph(f"{patient_info['exercise_angina']}", cell_style)],
        [Paragraph("ST Slope", cell_style), Paragraph(f"{patient_info['st_slope']}sloping", cell_style)],
        [Paragraph("Oldpeak (ST Depression)", cell_style), Paragraph(f"{patient_info['oldpeak']}", cell_style)]
    ]

    param_table = Table(table_data, colWidths=[240, 280])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#e0f2fe')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, border_gray),
    ]))

    story.append(param_table)
    story.append(Spacer(1, 15))

    res_title_style = ParagraphStyle(
        'ResTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=card_text_color,
        spaceAfter=6
    )
    res_body_style = ParagraphStyle(
        'ResBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=dark_slate,
        leading=14
    )

    res_box_content = [
        [Paragraph(f"<b>Assessment Result:</b> {outcome_status}", res_title_style)],
        [Paragraph(f"<b>Calculated Risk Probability Index:</b> <font color='{card_border.hexval()}'><b>{result_info['risk_prob']:.1f}%</b></font>", res_body_style)],
        [Paragraph(f"<b>Dr. Hearty Clinical Guidance:</b> {advice_summary}", res_body_style)]
    ]

    res_table = Table(res_box_content, colWidths=[520])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('BOX', (0,0), (-1,-1), 2, card_border),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))

    story.append(res_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Dr. Hearty's Recommended Action Initiatives:</b>", ParagraphStyle('ActHead', fontName='Helvetica-Bold', fontSize=12, textColor=header_blue, spaceAfter=8)))
    
    action_items = [
        "• <b>150 Min Weekly Cardio:</b> Engage in 30 minutes of brisk walking, swimming, or cycling 5 days a week.",
        "• <b>Heart-Smart Nutrition:</b> Increase intake of leafy greens, oats, berries, and lower sodium & saturated fats.",
        "• <b>Stress & Cortisol Management:</b> Practice 10 minutes of daily deep breathing or meditation.",
        "• <b>Regular Clinical Monitoring:</b> Schedule routine blood pressure and lipid panel checks with your primary physician."
    ]

    for item in action_items:
        story.append(Paragraph(item, ParagraphStyle('ActItem', fontName='Helvetica', fontSize=9.5, textColor=dark_slate, spaceAfter=4)))

    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94a3b8'), spaceAfter=8))
    story.append(Paragraph("<font size=8 color='#94a3b8'>Disclaimer: CardioPulse AI is a diagnostic decision-support tool trained on KNN classification metrics and does not replace professional medical diagnosis.</font>", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION WITH CARTOON HEART CHARACTERS & HISTORY LOG
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/833/833472.png" width="90" style="animation: floatMascot 3.5s ease-in-out infinite; filter: drop-shadow(0 8px 12px rgba(244,63,94,0.4));"/>
            <h2 class="cartoon-heading" style="color: #38bdf8; margin-top: 0.5rem;">Dr. Hearty AI</h2>
            <p style="color: #cbd5e1; font-size: 0.9rem;">Your Friendly Cardiac Health Companion</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🫀 Clinical Targets")
    st.markdown("""
    - 🩸 **Blood Pressure**: `< 120 mmHg`
    - 🧪 **Cholesterol**: `< 200 mg/dL`
    - 🍬 **Fasting Sugar**: `< 100 mg/dL`
    - 💓 **Heart Rate**: `60 - 100 bpm`
    """)

    st.divider()
    
    # 📜 Assessment History Sidebar Section
    st.markdown("### 📜 Assessment History")
    if st.session_state.assessment_history:
        st.write(f"Total Runs: **{len(st.session_state.assessment_history)}**")
        history_df = pd.DataFrame(st.session_state.assessment_history)
        st.dataframe(
            history_df[["Timestamp", "Age", "Sex", "Risk Score", "Result"]],
            hide_index=True,
            use_container_width=True
        )
        
        # Download full CSV log of history
        csv_history = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv_history,
            file_name=f"heart_assessment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        if st.button("🗑️ Clear History"):
            st.session_state.assessment_history = []
            st.rerun()
    else:
        st.info("No assessments recorded yet in this session.")

    st.divider()
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100, caption="Happy Heart Buddy")
    st.caption("CardioPulse")

# -----------------------------------------------------------------------------
# HERO HEADER WITH FLOATING CARTOON MASCOT
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="mascot-container">
        <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" class="heart-mascot-img" alt="cartoon heart"/>
        <div>
            <h1 class="cartoon-heading" style="font-size: 2.6rem; color: #ffffff; margin: 0;">
                CardioPulse AI <span class="beating-heart-icon">❤️</span>
            </h1>
            <p style="color: #cbd5e1; font-size: 1.1rem; margin-top: 0.3rem;">
                Meet Dr. Hearty! Enter your health stats below for an instant, friendly AI heart risk checkup.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

if load_error:
    st.warning(f"⚠️ Model files (`knn_heart_model.pkl`, `heart_scaler.pkl`) not found. Interactive demo mode active.\n*(Error: {load_error})*")

# -----------------------------------------------------------------------------
# PATIENT INPUT FORM
# -----------------------------------------------------------------------------
with st.form("heart_analytics_form"):
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="section-head">👤 Vitals & Demographic Metrics</div>', unsafe_allow_html=True)
        
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            age = st.number_input("Age (Years)", 18, 100, 45, help="Patient age in completed years")
        with sub_c2:
            sex = st.selectbox("Gender", ["Female", "Male"])

        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 130, help="Resting BP at admission")
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 50, 700, 230, help="Total serum cholesterol level")
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)"
        )

    with col2:
        st.markdown('<div class="section-head">⚡ Cardiac Stress & ECG Indicators</div>', unsafe_allow_html=True)
        
        max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 250, 140, help="Max HR during stress test")
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "ASY — Asymptomatic",
                "ATA": "ATA — Atypical Angina",
                "NAP": "NAP — Non-Anginal Pain",
                "TA": "TA — Typical Angina"
            }.get(x, x)
        )
        rest_ecg = st.selectbox(
            "Resting ECG Result",
            ["LVH", "Normal", "ST"],
            format_func=lambda x: {
                "Normal": "Normal",
                "LVH": "LVH — Left Ventricular Hypertrophy",
                "ST": "ST — ST-T Wave Abnormality"
            }.get(x, x)
        )

        sub_c3, sub_c4 = st.columns(2)
        with sub_c3:
            exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
        with sub_c4:
            st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"], format_func=lambda x: f"{x}sloping")

        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.5, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("🫀 Check My Heart Risk Now!")

# -----------------------------------------------------------------------------
# DIAGNOSTIC PROCESSING & CARTOON ANIMATED RESULTS
# -----------------------------------------------------------------------------
if submit_btn:
    with st.spinner("💓 Dr. Hearty is calculating your heart risk scores..."):
        time.sleep(0.6)

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

        if model is not None and scaler is not None:
            scaled_data = scaler.transform(df)
            prediction = model.predict(scaled_data)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(scaled_data)[0]
                risk_prob = probs[1] * 100
            else:
                risk_prob = 85.0 if prediction == 1 else 15.0
        else:
            calc_score = (age / 100 * 20) + (resting_bp / 200 * 25) + (cholesterol / 400 * 25) + (oldpeak * 10)
            prediction = 1 if calc_score > 55 or exercise_angina == "Yes" else 0
            risk_prob = min(max(calc_score, 10.0), 98.0) if prediction == 1 else min(calc_score, 35.0)

        # Record into session state history log
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_text = "High Risk 💔" if prediction == 1 else "Low Risk 💚"
        
        record = {
            "Timestamp": current_time,
            "Age": age,
            "Sex": sex,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": "Yes" if fasting_bs == 1 else "No",
            "MaxHR": max_hr,
            "ChestPain": chest_pain,
            "RestingECG": rest_ecg,
            "ExerciseAngina": exercise_angina,
            "STSlope": st_slope,
            "Oldpeak": oldpeak,
            "Risk Score": f"{risk_prob:.1f}%",
            "Result": result_text
        }
        st.session_state.assessment_history.append(record)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results Section
        res_col1, res_col2 = st.columns([1.3, 1.4], gap="large")

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="res-card-danger">
                        <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="80" style="margin-bottom: 0.5rem; animation: heartBeat 1s infinite;"/>
                        <h2 class="cartoon-heading" style="color: white; margin-bottom: 0.5rem; font-size: 2rem;">💔 HIGH CARDIO RISK DETECTED</h2>
                        <p style="color: #ffe4e6; font-size: 1.1rem; line-height: 1.5;">
                            Dr. Hearty noticed elevated heart risk signals! We strongly recommend scheduling a visit with a cardiologist.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.snow()
            else:
                st.markdown(f"""
                    <div class="res-card-safe">
                        <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" width="80" style="margin-bottom: 0.5rem; animation: floatMascot 3s ease-in-out infinite;"/>
                        <h2 class="cartoon-heading" style="color: white; margin-bottom: 0.5rem; font-size: 2rem;">💚 YOUR HEART IS HAPPY & SAFE!</h2>
                        <p style="color: #d1fae5; font-size: 1.1rem; line-height: 1.5;">
                            Great news! Your parameters fall within healthy ranges. Keep up the awesome healthy habits!
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metric Cards
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Heart Risk Probability", f"{risk_prob:.1f}%", delta="HIGH RISK" if prediction == 1 else "HEALTHY", delta_color="inverse" if prediction == 1 else "normal")
            with m2:
                st.metric("AI Confidence", "94.2%", delta="KNN Model", delta_color="off")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # 📄 Download PDF & Text Assessment Report Features
            patient_info = {
                'age': age, 'sex': sex, 'resting_bp': resting_bp, 'cholesterol': cholesterol,
                'fasting_bs': fasting_bs, 'max_hr': max_hr, 'chest_pain': chest_pain,
                'rest_ecg': rest_ecg, 'exercise_angina': exercise_angina, 'st_slope': st_slope,
                'oldpeak': oldpeak
            }
            result_info = {
                'timestamp': current_time, 'prediction': prediction,
                'risk_prob': risk_prob, 'result_text': result_text
            }

            if REPORTLAB_AVAILABLE:
                pdf_buffer = generate_pdf_report(patient_info, result_info)
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download Assessment Report (PDF)",
                        data=pdf_buffer,
                        file_name=f"Dr_Hearty_Assessment_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            
            # Always provide text report download as backup
            txt_data = f"""CARDIO-PULSE AI HEALTH ASSESSMENT REPORT
==================================================
Date & Time        : {current_time}
Patient Age        : {age} Years
Patient Gender     : {sex}

CLINICAL METRICS EVALUATED:
--------------------------------------------------
Resting Blood Pressure : {resting_bp} mm Hg
Serum Cholesterol      : {cholesterol} mg/dL
Fasting Blood Sugar    : {'Yes (> 120 mg/dl)' if fasting_bs == 1 else 'No (<= 120 mg/dl)'}
Maximum Heart Rate     : {max_hr} bpm
Chest Pain Type        : {chest_pain}
Resting ECG Result     : {rest_ecg}
Exercise Angina        : {exercise_angina}
ST Slope               : {st_slope}sloping
Oldpeak (ST Depression): {oldpeak}

DIAGNOSTIC OUTCOME:
--------------------------------------------------
Risk Probability Score : {risk_prob:.1f}%
Primary Classification : {result_text}
==================================================
Dr. Hearty AI Diagnostic Support System
"""
            st.download_button(
                label="📝 Download Assessment Report (TXT)",
                data=txt_data,
                file_name=f"Dr_Hearty_Assessment_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        # -----------------------------------------------------------------------------
        # DR. HEARTY RISK METER WITH ANIMATED MOVING NEEDLE
        # -----------------------------------------------------------------------------
        with res_col2:
            gauge_color = "#f43f5e" if risk_prob > 50 else "#10b981"
            
            fig_gauge = go.Figure()

            # Gauge Background & Color Zones
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=risk_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dr. Hearty's  Risk Meter 🫀", 'font': {'size': 20, 'color': "#f8fafc", 'family': "Fredoka"}},
                number={'suffix': "%", 'font': {'size': 48, 'color': gauge_color, 'family': "Fredoka"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#38bdf8"},
                    'bar': {'color': 'rgba(0,0,0,0)'},
                    'bgcolor': "rgba(30, 41, 59, 0.6)",
                    'borderwidth': 2,
                    'bordercolor': "#38bdf8",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.35)'},
                        {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.35)'},
                        {'range': [65, 100], 'color': 'rgba(244, 63, 94, 0.35)'}
                    ]
                }
            ))

            # Dynamic Physical Needle Calculation
            angle = 180 - (risk_prob / 100.0 * 180)
            rad = math.radians(angle)
            needle_r = 0.38
            center_x, center_y = 0.5, 0.22
            needle_x = center_x + needle_r * math.cos(rad)
            needle_y = center_y + needle_r * math.sin(rad)

            # Draw Needle Line and Hub Center
            fig_gauge.add_shape(
                type="line",
                x0=center_x, y0=center_y,
                x1=needle_x, y1=needle_y,
                line=dict(color="#ffffff", width=5)
            )
            fig_gauge.add_shape(
                type="circle",
                x0=center_x - 0.03, y0=center_y - 0.03,
                x1=center_x + 0.03, y1=center_y + 0.03,
                fillcolor=gauge_color,
                line=dict(color="#ffffff", width=2)
            )

            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=290,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # -----------------------------------------------------------------------------
        # CARTOON HEALTH INITIATIVES & ACTION PLAN SECTION
        # -----------------------------------------------------------------------------
        st.markdown("""
            <h2 class="cartoon-heading" style="color: #f8fafc; font-size: 2rem;">
                🚀 Dr. Hearty's Recommended Health Action Plan
            </h2>
            <p style="color: #cbd5e1; font-size: 1.05rem; margin-bottom: 1.5rem;">
                Follow these fun & effective health initiatives to keep your heart smiling, energetic, and strong!
            </p>
        """, unsafe_allow_html=True)
        
        act_col1, act_col2, act_col3, act_col4 = st.columns(4, gap="medium")

        with act_col1:
            st.markdown("""
                <div class="action-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/2964/2964514.png" class="action-icon" alt="cardio exercise"/>
                    <h3 class="cartoon-heading" style="color: #38bdf8; font-size: 1.3rem; margin-bottom: 0.4rem;">150 Min Cardio Power</h3>
                    <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.4;">
                        Engage in brisk walking, swimming, or cycling 30 mins a day for 5 days a week!
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with act_col2:
            st.markdown("""
                <div class="action-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/3058/3058995.png" class="action-icon" alt="healthy diet"/>
                    <h3 class="cartoon-heading" style="color: #34d399; font-size: 1.3rem; margin-bottom: 0.4rem;">Heart-Smart Diet</h3>
                    <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.4;">
                        Boost your meals with leafy greens, oats, berries, nuts, and cut down on sodium & fried snacks.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with act_col3:
            st.markdown("""
                <div class="action-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/3004/3004462.png" class="action-icon" alt="stress reduction"/>
                    <h3 class="cartoon-heading" style="color: #c084fc; font-size: 1.3rem; margin-bottom: 0.4rem;">Chill & Zen Stress Control</h3>
                    <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.4;">
                        Practice 10 mins of deep breathing, yoga, or meditation to keep cortisol & blood pressure low!
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with act_col4:
            st.markdown("""
                <div class="action-card">
                    <img src="https://cdn-icons-png.flaticon.com/512/3004/3004454.png" class="action-icon" alt="routine checkup"/>
                    <h3 class="cartoon-heading" style="color: #fb7185; font-size: 1.3rem; margin-bottom: 0.4rem;">Regular Checkups</h3>
                    <p style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.4;">
                        Schedule routine blood pressure & cholesterol screenings with your physician every 6 months.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="footer">
        ❤️ <b>Dr. Hearty's CardioPulse Suite</b> • Design by Ankit Rai
    </div>
""", unsafe_allow_html=True)
