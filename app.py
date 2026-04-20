import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Student Dashboard", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("student-scores.csv")
    data = data.drop(["id", "first_name", "last_name", "email"], axis=1, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    return data

data = load_data()

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = data[[
    "weekly_self_study_hours",
    "absence_days",
    "part_time_job",
    "extracurricular_activities"
]]

y = data[[
    "math_score",
    "physics_score",
    "chemistry_score",
    "biology_score",
    "english_score",
    "history_score",
    "geography_score"
]]

# -------------------------------
# MODEL
# -------------------------------
@st.cache_resource
def train_model(X, y):
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

# -------------------------------
# PDF GENERATOR
# -------------------------------
def generate_pdf(name, study_hours, absence, part_time, activities, prediction):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("<b>Student Performance Report</b>", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Study Hours: {study_hours}", styles["Normal"]))
    content.append(Paragraph(f"Absence Days: {absence}", styles["Normal"]))
    content.append(Paragraph(f"Part Time Job: {part_time}", styles["Normal"]))
    content.append(Paragraph(f"Extracurricular Activities: {activities}", styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Predicted Scores:</b>", styles["Heading2"]))

    subjects = ["Math", "Physics", "Chemistry", "Biology", "English", "History", "Geography"]

    for sub, score in zip(subjects, prediction):
        content.append(Paragraph(f"{sub}: {round(score, 2)}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict scores and download report</p>", unsafe_allow_html=True)

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📥 Enter Student Details")

name = st.text_input("👤 Student Name")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Weekly Study Hours", 0, 50, 10)
    part_time = st.selectbox("Part Time Job", [False, True])

with col2:
    absence = st.slider("Absence Days", 0, 10, 2)
    activities = st.selectbox("Extracurricular Activities", [False, True])

st.divider()

# -------------------------------
# BUTTON
# -------------------------------
center = st.columns([1, 2, 1])
with center[1]:
    predict_btn = st.button("🚀 Predict & Generate Report", use_container_width=True)

# -------------------------------
# PREDICTION
# -------------------------------
if predict_btn:

    if name.strip() == "":
        st.warning("⚠️ Please enter student name")
    else:
        input_data = [[study_hours, absence, part_time, activities]]
        prediction = model.predict(input_data)[0]

        subjects = [
            "Math", "Physics", "Chemistry",
            "Biology", "English", "History", "Geography"
        ]

        st.divider()
        st.subheader(f"📊 Prediction for {name}")

        # -------------------------------
        # METRICS
        # -------------------------------
        cols = st.columns(4)
        for i, subject in enumerate(subjects):
            cols[i % 4].metric(subject, round(prediction[i], 2))

        # -------------------------------
        # GRAPH (INTERACTIVE)
        # -------------------------------
        pred_df = pd.DataFrame({
            "Subject": subjects,
            "Score": prediction
        })

        st.subheader("📈 Score Trend")

        fig = px.line(pred_df, x="Subject", y="Score", markers=True)
        fig.update_traces(line=dict(width=4), marker=dict(size=8))
        fig.update_layout(yaxis=dict(range=[0, 100]))

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # SAVE TO CSV
        # -------------------------------
        save_data = pd.DataFrame({
            "name": [name],
            "study_hours": [study_hours],
            "absence": [absence],
            "part_time": [part_time],
            "activities": [activities],
            "math": [prediction[0]],
            "physics": [prediction[1]],
            "chemistry": [prediction[2]],
            "biology": [prediction[3]],
            "english": [prediction[4]],
            "history": [prediction[5]],
            "geography": [prediction[6]]
        })

        file_path = "predictions_log.csv"
        file_exists = os.path.isfile(file_path)

        save_data.to_csv(file_path, mode='a', header=not file_exists, index=False)

        st.success("✅ Data saved successfully!")

        # -------------------------------
        # PDF DOWNLOAD
        # -------------------------------
        pdf_file = generate_pdf(name, study_hours, absence, part_time, activities, prediction)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name=f"{name}_report.pdf",
            mime="application/pdf"
        )

        # -------------------------------
        # INSIGHT
        # -------------------------------
        avg_score = pred_df["Score"].mean()

        if avg_score > 80:
            st.success("🎯 Excellent performance expected")
        elif avg_score > 60:
            st.info("📊 Average performance")
        else:
            st.warning("⚠️ Needs improvement")