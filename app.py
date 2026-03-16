import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction Dashboard")

st.write("Machine Learning project to predict student final scores.")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():

    path = os.path.join(os.path.dirname(__file__), "student_performance_updated_1000.csv")

    if not os.path.exists(path):
        st.error("Dataset not found")
        return None

    data = pd.read_csv(path)

    required_cols = ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "FinalGrade"]
    for col in required_cols:
        if col not in data.columns:
            st.error(f"Required column missing: {col}")
            return None

    # Fill missing values for model features/target with column mean
    data[required_cols] = data[required_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
    data[required_cols] = data[required_cols].fillna(data[required_cols].mean())

    return data


df = load_data()

# -----------------------------
# MAIN DASHBOARD
# -----------------------------
if df is not None:

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Dataset",
            "Visualization",
            "Model Accuracy",
            "Prediction"
        ]
    )

# -----------------------------
# DATASET PAGE
# -----------------------------
    if page == "Dataset":

        st.header("Dataset Preview")

        st.write(df.head())

        st.write("Dataset Shape:", df.shape)

# -----------------------------
# VISUALIZATION PAGE
# -----------------------------
    if page == "Visualization":

        st.header("Data Visualization")

        fig, ax = plt.subplots()

        ax.scatter(
            df["StudyHoursPerWeek"],
            df["FinalGrade"]
        )

        ax.set_xlabel("Study Hours")

        ax.set_ylabel("Final Grade")

        st.pyplot(fig)

# -----------------------------
# MODEL TRAINING
# -----------------------------
    features = [
        "AttendanceRate",
        "StudyHoursPerWeek",
        "PreviousGrade"
    ]

    target = "FinalGrade"

    X = df[features]

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

# -----------------------------
# MODEL ACCURACY
# -----------------------------
    if page == "Model Accuracy":

        st.header("Model Performance")

        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)

        st.metric(
            label="R2 Score",
            value=round(r2, 3)
        )

# -----------------------------
# PREDICTION PAGE
# -----------------------------
    if page == "Prediction":

        st.header("Predict Student Final Score")

        attendance = st.slider(
            "Attendance Rate (%)",
            0,
            100,
            85
        )

        study_hours = st.slider(
            "Study Hours per Week",
            0,
            50,
            15
        )

        previous_grade = st.slider(
            "Previous Grade (%)",
            0,
            100,
            70
        )

        if st.button("Predict Score"):

            input_data = np.array([
                [
                    attendance,
                    study_hours,
                    previous_grade
                ]
            ])

            prediction = model.predict(input_data)

            st.success(
                f"Predicted Final Score: {prediction[0]:.2f}"
            )

            st.balloons()

else:

    st.warning("Dataset not available.")