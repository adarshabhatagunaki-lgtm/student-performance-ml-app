import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction Dashboard")
st.write("Machine Learning project to predict student final scores")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():

    path = "C:\Users\adarsh\Documents\student_performance_1000.csv"

    if not os.path.exists(path):
        st.error("Dataset not found in project folder")
        return None

    df = pd.read_csv(path)

    numeric_cols = [
        "AttendanceRate",
        "StudyHoursPerWeek",
        "PreviousGrade",
        "FinalGrade"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df


df = load_data()


# -----------------------------
# MODEL TRAINING
# -----------------------------
if df is not None:

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

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        [
            "Dashboard",
            "Dataset",
            "Visualization",
            "Model Accuracy",
            "Prediction"
        ]
    )


# -----------------------------
# DASHBOARD PAGE
# -----------------------------
    if page == "Dashboard":

        st.header("📊 Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", len(df))
        col2.metric("Average Final Grade", round(df["FinalGrade"].mean(), 2))
        col3.metric("Model R² Score", round(r2, 3))

        st.subheader("Final Grade Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["FinalGrade"], bins=20)

        ax.set_xlabel("Final Grade")
        ax.set_ylabel("Number of Students")

        st.pyplot(fig)


# -----------------------------
# DATASET PAGE
# -----------------------------
    if page == "Dataset":

        st.header("Dataset Preview")

        st.dataframe(df)

        st.write("Dataset Shape:", df.shape)


# -----------------------------
# VISUALIZATION PAGE
# -----------------------------
    if page == "Visualization":

        st.header("Study Hours vs Final Grade")

        fig1, ax1 = plt.subplots()

        ax1.scatter(
            df["StudyHoursPerWeek"],
            df["FinalGrade"]
        )

        ax1.set_xlabel("Study Hours Per Week")
        ax1.set_ylabel("Final Grade")

        st.pyplot(fig1)


        st.header("Attendance Rate vs Final Grade")

        fig2, ax2 = plt.subplots()

        ax2.scatter(
            df["AttendanceRate"],
            df["FinalGrade"]
        )

        ax2.set_xlabel("Attendance Rate")
        ax2.set_ylabel("Final Grade")

        st.pyplot(fig2)


# -----------------------------
# MODEL ACCURACY PAGE
# -----------------------------
    if page == "Model Accuracy":

        st.header("Model Performance")

        st.metric("R² Score", round(r2, 3))

        st.subheader("Actual vs Predicted Scores")

        fig3, ax3 = plt.subplots()

        ax3.scatter(y_test, predictions)

        ax3.set_xlabel("Actual Score")
        ax3.set_ylabel("Predicted Score")

        st.pyplot(fig3)


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

            if prediction >= 85:
                st.success("Excellent Performance 🎉")

            elif prediction >= 70:
                st.info("Good Performance 👍")

            elif prediction >= 50:
                st.warning("Average Performance")

            else:
                st.error("Needs Improvement ⚠")

            st.balloons()

else:

    st.warning("Dataset not available.")