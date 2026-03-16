import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -----------------------------
# Load Dataset
# -----------------------------
def load_data():
    # Prefer local project path so code works on any machine.
    local_path = os.path.join(os.path.dirname(__file__), "student_performance_updated_1000.csv")
    fallback_path = "C:/Users/adarsh/Desktop/student_project/student_performance_updated_1000.csv"

    if os.path.exists(local_path):
        path = local_path
    elif os.path.exists(fallback_path):
        path = fallback_path
    else:
        raise FileNotFoundError(
            f"Dataset not found. Checked: {local_path} and {fallback_path}"
        )

    data = pd.read_csv(path)
    return data


# -----------------------------
# Data Exploration
# -----------------------------
def explore_data(df):

    print("First 5 rows")
    print(df.head())

    print("\nDataset Info")
    print(df.info())

    print("\nStatistics")
    print(df.describe())


# -----------------------------
# Feature Selection
# -----------------------------
def split_features(df):

    X = df[
        [
            "study_hours",
            "previous_score",
            "sleep_hours",
            "practice_papers",
            "activities"
        ]
    ]

    y = df["final_score"]

    return X, y


# -----------------------------
# Train Test Split
# -----------------------------
def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# Train Model
# -----------------------------
def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


# -----------------------------
# Evaluate Model
# -----------------------------
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# -----------------------------
# Save Model
# -----------------------------
def save_model(model):

    os.makedirs("models", exist_ok=True)

    pickle.dump(
        model,
        open("models/model.pkl", "wb")
    )

    print("Model saved successfully")


# -----------------------------
# Main Function
# -----------------------------
def main():

    df = load_data()

    explore_data(df)

    X, y = split_features(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()