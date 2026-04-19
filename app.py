import streamlit as st
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------
# Load Data
# -----------------------
data = pd.read_csv("student-scores.csv")
data = data.drop(["id", "first_name", "last_name", "email"], axis=1, errors='ignore')

# Encode categorical
data = pd.get_dummies(data, drop_first=True)

# -----------------------
# Features (ONLY BEHAVIORAL)
# -----------------------
X = data[[
    "weekly_self_study_hours",
    "absence_days",
    "part_time_job",
    "extracurricular_activities"
]]

# -----------------------
# Targets (ALL SUBJECTS)
# -----------------------
y = data[[
    "math_score",
    "physics_score",
    "chemistry_score",
    "biology_score",
    "english_score",
    "history_score",
    "geography_score"
]]

# -----------------------
# Train Model
# -----------------------
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42)
)
model.fit(X, y)

# -----------------------
# UI
# -----------------------
st.title("🎓 Student Score Prediction Dashboard")

st.write("Enter student details to predict all subject scores")

# Inputs (ONLY behavioral)
study_hours = st.slider("Weekly Study Hours", 0, 50, 10)
absence = st.slider("Absence Days", 0, 10, 2)
part_time = st.selectbox("Part Time Job", [False, True])
activities = st.selectbox("Extracurricular Activities", [False, True])

# Predict Button
if st.button("Predict Scores"):
    
    input_data = [[
        study_hours,
        absence,
        part_time,
        activities
    ]]
    
    prediction = model.predict(input_data)

    math, physics, chemistry, biology, english, history, geography = prediction[0]

    st.success("📊 Predicted Scores:")

    st.write(f"Math: {round(math, 2)}")
    st.write(f"Physics: {round(physics, 2)}")
    st.write(f"Chemistry: {round(chemistry, 2)}")
    st.write(f"Biology: {round(biology, 2)}")
    st.write(f"English: {round(english, 2)}")
    st.write(f"History: {round(history, 2)}")
    st.write(f"Geography: {round(geography, 2)}")

# -----------------------
# Dataset Preview
# -----------------------
st.subheader("Dataset Preview")
st.write(data.head())