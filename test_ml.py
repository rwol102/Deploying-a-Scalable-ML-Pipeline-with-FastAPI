import pytest
import pandas as pd
import numpy as np
from funcy import none_fn
from scipy.constants import precision
from sklearn.metrics import fbeta_score
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics
)
from train_model import cat_features
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_shapes():
    sample_data = pd.DataFrame({
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Sales"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "United-States"],
        "salary": [">50K", "<=50K"]
    })

    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == y.shape[0]
    assert len(X.shape) == 2


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_model():
    X = np.array([[0,1],[1,0]])
    y = np.array([1,0])

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_range():
    y_true = np.array([0,1,1,0])
    y_pred = np.array([0,1,1,0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
