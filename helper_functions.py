import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Dataset Summary
# ---------------------------
def dataset_summary(df):

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum()
    }

    return summary


# ---------------------------
# Correlation Plot
# ---------------------------
def correlation_plot(df):

    corr = df.corr()

    plt.figure(figsize=(8,6))

    plt.imshow(corr)

    plt.colorbar()

    plt.title("Feature Correlation")

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()


# ---------------------------
# Study Hours vs Score
# ---------------------------
def plot_study_vs_score(df):

    plt.scatter(df["study_hours"], df["final_score"])

    plt.xlabel("Study Hours")

    plt.ylabel("Final Score")

    plt.title("Study Hours vs Score")

    plt.show()