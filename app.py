import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model = pickle.load(open("knn_heart_model.pkl","rb"))
scaler = pickle.load(open("heart_scaler.pkl","rb"))

st.markdown("""
<style>

.stApp{
background:#edf5f3;
}

.main{
padding-top:0rem;
}

.block-container{
padding-top:1rem;
padding-bottom:1rem;
}

.hero{

background:#071d18;
padding:35px;
border-radius:18px;
color:white;

box-shadow:0px 8px 20px rgba(0,0,0,.25);

}

.ecg{

font-size:24px;
color:#3fd1c2;

letter-spacing:3px;

}

.section{

background:white;

padding:25px;

border-radius:15px;

margin-top:20px;

box-shadow:0px 3px 10px rgba(0,0,0,.08);

}

.result{

background:white;

padding:25px;

border-radius:15px;

text-align:center;

box-shadow:0px 3px 10px rgba(0,0,0,.1);

}

div.stButton>button{

background:#06261f;

color:white;

font-size:22px;

height:60px;

border-radius:12px;

width:100%;

font-weight:bold;

}

div.stButton>button:hover{

background:#0b5f52;

color:white;

}

</style>
""",unsafe_allow_html=True)

st.markdown("""

<div class="hero">

<h5 style="letter-spacing:4px;color:#65d7cb;">
AI POWERED CLINICAL SUPPORT
</h5>

<h1>
❤️ Heart Disease Prediction
</h1>

<p style="font-size:20px;">
Predict heart disease risk using Machine Learning.
</p>

<div class="ecg">

──────────────────╱╲────────╱╲────────────────

</div>

</div>

""",unsafe_allow_html=True)

left,right=st.columns([2,1])

with left:

    st.markdown(
    '<div class="section">',
    unsafe_allow_html=True)

    st.subheader("👤 Patient Profile")

    st.write("Fill patient information.")

    st.markdown("</div>",unsafe_allow_html=True)

with right:

    st.markdown(
    '<div class="result">',
    unsafe_allow_html=True)

    st.subheader("❤️ Prediction Result")

    st.info("Fill all patient details and press **Run Diagnosis**.")

    st.markdown("</div>",unsafe_allow_html=True)

with left:

    st.markdown("## 👤 Patient Profile")

    c1,c2=st.columns(2)

    with c1:
        age=st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=45
        )

    with c2:
        sex=st.selectbox(
            "Gender",
            ["Male","Female"]
        )

    c1,c2=st.columns(2)

    with c1:

        chest_pain=st.selectbox(
            "Chest Pain Type",
            [
                "ATA",
                "NAP",
                "TA",
                "ASY"
            ]
        )

    with c2:

        resting_ecg=st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "LVH",
                "ST"
            ]
        )

    st.markdown("---")
    st.markdown("## 🩺 Vitals & Laboratory")

    c1,c2=st.columns(2)

    with c1:

        resting_bp=st.number_input(
            "Resting Blood Pressure",
            50,
            250,
            120
        )

    with c2:

        cholesterol=st.number_input(
            "Cholesterol",
            0,
            700,
            200
        )

    c1,c2=st.columns(2)

    with c1:

        max_hr=st.number_input(
            "Maximum Heart Rate",
            50,
            250,
            150
        )

    with c2:

        oldpeak=st.number_input(
            "Old Peak",
            0.0,
            10.0,
            1.0,
            step=0.1
        )

    fasting_bs=st.checkbox(
        "Fasting Blood Sugar > 120 mg/dl"
    )

    st.markdown("---")
    st.markdown("## 📈 Cardiac Test Results")

    c1,c2=st.columns(2)

    with c1:

        exercise_angina=st.selectbox(
            "Exercise Angina",
            [
                "No",
                "Yes"
            ]
        )

    with c2:

        st_slope=st.selectbox(
            "ST Slope",
            [
                "Up",
                "Flat",
                "Down"
            ]
        )

st.markdown("<br>",unsafe_allow_html=True)

predict=st.button(
    "❤️ Run Diagnosis"
)

with right:

    st.markdown("""
    <div class="result">

    <h2>❤️ Prediction Panel</h2>

    <br>

    <h1>🫀</h1>

    <br>

    Fill patient details and click
    <br><br>

    <b>Run Diagnosis</b>

    </div>
    """,unsafe_allow_html=True)

predict = st.button("❤️ Run Diagnosis")

if predict:

    sex_m = 1 if sex == "Male" else 0

    chest_ata = 1 if chest_pain == "ATA" else 0
    chest_nap = 1 if chest_pain == "NAP" else 0
    chest_ta = 1 if chest_pain == "TA" else 0

    ecg_normal = 1 if resting_ecg == "Normal" else 0
    ecg_st = 1 if resting_ecg == "ST" else 0

    exercise_y = 1 if exercise_angina == "Yes" else 0

    slope_flat = 1 if st_slope == "Flat" else 0
    slope_up = 1 if st_slope == "Up" else 0

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

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]


with right:

    st.markdown("## ❤️ Prediction Result")

    if predict:

        if prediction == 1:

            st.error("🚨 High Risk of Heart Disease")

            st.progress(int(probability*100))

            st.metric(
                "Risk Score",
                f"{probability*100:.1f}%"
            )

            st.warning(
                "Please consult a cardiologist for further evaluation."
            )

            st.snow()

        else:

            st.success("🎉 Low Risk of Heart Disease")

            st.progress(int((1-probability)*100))

            st.metric(
                "Heart Health",
                f"{(1-probability)*100:.1f}%"
            )

            st.success(
                "Maintain a healthy lifestyle!"
            )

            st.balloons()

    else:

        st.info("Fill all details and click **Run Diagnosis**.")

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966485.png",
        width=100
    )

    st.title("Heart AI")

    st.markdown("---")

    st.info(
        """
        **Heart Disease Prediction**

        ✔ Machine Learning

        ✔ KNN Classifier

        ✔ Clinical Support

        ✔ Instant Prediction
        """
    )

    st.markdown("---")

    st.success("Version 1.0")

m1,m2,m3,m4=st.columns(4)

m1.metric(
    "Patients",
    "918"
)

m2.metric(
    "Model",
    "KNN"
)

m3.metric(
    "Accuracy",
    "87%"
)

m4.metric(
    "Status",
    "Online"
)

with st.expander("👤 Patient Profile",expanded=True):

    # age

    # gender

    # chest pain

    # ecg

with st.expander("🩺 Vital Signs",expanded=True):

    # bp

    # cholesterol

    # fasting

with st.expander("📈 Cardiac Tests",expanded=True):

    # maxhr

    # oldpeak

    # slope

    # angina

if predict:

    probability=model.predict_proba(scaled_data)[0][1]

    st.write("### Risk Probability")

    st.progress(probability)

    st.write(f"### {probability*100:.1f}%")

st.success("""

### Recommendation

🥗 Healthy Diet

🏃 Regular Exercise

😴 Sleep 7–8 Hours

🩺 Annual Checkup

""")

st.error("""

### Recommendation

🚑 Visit Cardiologist

💊 Follow Doctor Advice

🩺 ECG Test

🧪 Blood Test

❤️ Lifestyle Changes

""")

st.markdown("---")

st.markdown(
"""
<center>

### ❤️ Heart Disease Prediction

Developed using

Python • Streamlit • Scikit-Learn

© 2026 Ankit Rai

</center>

""",
unsafe_allow_html=True
)

with st.spinner("Analyzing patient data..."):

    scaled=scaler.transform(input_data)

    prediction=model.predict(scaled)

st.balloons()

st.toast("Patient appears to be at low risk ❤️")


st.toast("High Risk Detected 🚨")

st.markdown("## Patient Summary")

summary=pd.DataFrame({

"Feature":[

"Age",

"Blood Pressure",

"Cholesterol",

"Max HR"

],

"Value":[

age,

resting_bp,

cholesterol,

max_hr

]

})

st.table(summary)

