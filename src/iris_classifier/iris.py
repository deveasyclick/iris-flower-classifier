import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
x = pd.read_csv(DATA_PATH)

y = x["species"]

x.drop(columns=["species"], inplace=True)


# Split data into test and train dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Scale down the vaules to fit between 0 and 1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create model
model = LogisticRegression()

# Cross validation
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# Train model
# trained_model = model.fit(x_train, y_train)


# Predict
prediction = best_model.predict(x_test)


# Evaluate
accuracy = accuracy_score(y_test, prediction)

# Confusion matrix
matrix = confusion_matrix(y_test, prediction)

# Report
report = classification_report(y_test, prediction)

print(f"Accuracy score: {accuracy}")
print(f"Confusion matrix: {matrix}")
print(f"Report: {report}")
print(y.value_counts())
print(f"Best model: {best_model}")
print(f"Best score: {grid_search.best_score_}")


def plot_model(matrix):
    plt.figure(figsize=(10, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()


plot_model(matrix)
