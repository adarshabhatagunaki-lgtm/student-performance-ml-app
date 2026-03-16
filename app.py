import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")
st.markdown("Enter the details below to predict the student's final score.")

# --- 2. DATA LOADING (Safe Version) ---
def load_data():
    # Use a relative path so it works on both your PC and the Cloud
    path = "C:\\Users\\adarsh\\Desktop\\student-performance-ml-app\\student_performance_updated_1000.csv"
    
    # If the file is not in the main folder, check the 'dataset' folder
    if not os.path.exists(path):
        path = os.path.join("dataset", path)
        
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # Instead of exit(), we show an error on the screen
        st.error(f"⚠️ Dataset not found at: {path}")
        return None

df = load_data()
if df is not None:
    try:
        # Training a simple model on the fly for the app
        # The dataset has these fields: AttendanceRate, StudyHoursPerWeek, PreviousGrade, FinalGrade
        required_features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade']
        target_column = 'FinalGrade'

        missing = [c for c in required_features + [target_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        X = df[required_features]
        y = df[target_column]

        model = LinearRegression()
        model.fit(X, y)

        # --- USER INPUT SECTION ---
        st.divider()
        st.subheader("Student Metrics")

        attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
        study = st.number_input("Study Hours per Week", 0, 100, 15)
        previous_grade = st.slider("Previous Grade (%)", 0, 100, 75)

        # --- PREDICTION SECTION ---
        st.divider()
        if st.button("Calculate Predicted Score", type="primary"):
            input_data = np.array([[attendance, study, previous_grade]])
            prediction = model.predict(input_data)
            
            # Show the result in a nice box
            st.balloons()
            st.success(f"### Predicted Final Score: {prediction[0]:.2f}")
            
    except Exception as e:
        st.warning(f"Error setting up the model: {e}")
        st.info("Check if your CSV column names match: 'AttendanceRate', 'StudyHoursPerWeek','PreviousGrade', and 'FinalGrade'.")

else:
    st.info("Please make sure your CSV file is uploaded to your GitHub repository.")