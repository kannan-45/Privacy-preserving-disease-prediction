"""
Data loading, preprocessing, and non-IID hospital splitting for the
federated learning pipeline.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

DIABETES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"


def load_datasets():
    """Load the Diabetes and Breast Cancer datasets as DataFrames."""
    diabetes = pd.read_csv(DIABETES_URL)

    data = load_breast_cancer()
    cancer = pd.DataFrame(data.data, columns=data.feature_names)
    cancer["target"] = data.target

    return diabetes, cancer


def preprocess(df, target):
    """Scale features and split into stratified train/test sets."""
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


def pad(X, max_features):
    """Zero-pad feature matrix so all datasets share the same width."""
    if X.shape[1] < max_features:
        return np.hstack((X, np.zeros((X.shape[0], max_features - X.shape[1]))))
    return X


def to_tensor(X, y):
    """Convert numpy arrays to torch tensors."""
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).view(-1, 1),
    )


def split_hospitals_noniid(X, y):
    """Split a dataset into 3 non-IID hospital partitions by class ratio."""
    y_flat = y.view(-1)

    idx0 = (y_flat == 0).nonzero(as_tuple=True)[0]
    idx1 = (y_flat == 1).nonzero(as_tuple=True)[0]

    idx0 = idx0[torch.randperm(len(idx0))]
    idx1 = idx1[torch.randperm(len(idx1))]

    hospitals = [
        (X[torch.cat([idx0[:200], idx1[:50]])], y[torch.cat([idx0[:200], idx1[:50]])]),
        (X[torch.cat([idx0[200:250], idx1[50:200]])], y[torch.cat([idx0[200:250], idx1[50:200]])]),
        (X[torch.cat([idx0[250:350], idx1[200:300]])], y[torch.cat([idx0[250:350], idx1[200:300]])]),
    ]

    return hospitals


def prepare_data():
    """Full pipeline: load, preprocess, pad, tensorize, and split into hospitals.

    Returns:
        hospitals: list of (X, y) tensor pairs, one per simulated hospital
        test_sets: dict with 'diabetes', 'cancer', and 'combined' (X_test, y_test) tensors
        max_features: the padded feature width used by the model
    """
    diabetes, cancer = load_datasets()

    X1_train, X1_test, y1_train, y1_test = preprocess(diabetes, "Outcome")
    X2_train, X2_test, y2_train, y2_test = preprocess(cancer, "target")

    max_features = max(X1_train.shape[1], X2_train.shape[1])

    X1_train, X1_test = pad(X1_train, max_features), pad(X1_test, max_features)
    X2_train, X2_test = pad(X2_train, max_features), pad(X2_test, max_features)

    X1_train, y1_train = to_tensor(X1_train, y1_train)
    X1_test, y1_test = to_tensor(X1_test, y1_test)

    X2_train, y2_train = to_tensor(X2_train, y2_train)
    X2_test, y2_test = to_tensor(X2_test, y2_test)

    hospitals = split_hospitals_noniid(X1_train, y1_train) + \
        split_hospitals_noniid(X2_train, y2_train)

    test_sets = {
        "diabetes": (X1_test, y1_test),
        "cancer": (X2_test, y2_test),
        "combined": (torch.cat((X1_test, X2_test)), torch.cat((y1_test, y2_test))),
    }

    return hospitals, test_sets, max_features
