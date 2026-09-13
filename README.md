# Iris Classifier

A small machine learning project that classifies Iris flower species using a
Logistic Regression model trained with scikit-learn.

The dataset contains 150 samples across three species — *Iris-setosa*,
*Iris-versicolor*, and *Iris-virginica* — each described by four features:
sepal length, sepal width, petal length, and petal width.

## Features

- Loads the Iris dataset from a bundled CSV file
- Splits data into training (75%) and test (25%) sets
- Scales features to the `[0, 1]` range with `MinMaxScaler`
- Tunes the regularization strength `C` using `GridSearchCV` with 5-fold cross-validation
- Evaluates the best model with accuracy, a confusion matrix, and a classification report
- Plots the confusion matrix as a heatmap with seaborn/matplotlib

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) (recommended)

Dependencies are managed in `pyproject.toml`:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Setup

Clone the repository and install the dependencies with uv:

```bash
git clone git@github.com:deveasyclick/iris-flower-classifier.git
cd iris-classifier
uv sync
```

## Usage

Run the classifier script:

```bash
uv run src/iris_classifier/iris.py
```

You can also run the installed package entry point:

```bash
uv run iris-classifier
```

### Output

The script prints:

- Accuracy score on the test set
- Confusion matrix
- Full classification report (precision, recall, F1-score)
- Sample counts per species
- The best estimator and its cross-validation score

It also opens an interactive window with the confusion matrix heatmap.

## Project Structure

```
iris-classifier/
├── pyproject.toml
├── uv.lock
├── README.md
└── src/
    └── iris_classifier/
        ├── __init__.py
        ├── iris.py            # Training, tuning, and evaluation script
        └── data/
            └── iris.csv       # Iris dataset
```

## Dataset

`src/iris_classifier/data/iris.csv` has the following columns:

| Column | Description |
| --- | --- |
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |
| `species` | Target class (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) |

## License

This project is for educational purposes.
