"""
Evaluation metrics and communication-cost utilities.
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, X, y):
    """Compute Accuracy, Precision, Recall, and F1 for a model on (X, y)."""
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).float()

    y_true = y.numpy()
    y_pred = preds.numpy()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def model_size(model):
    """Return (param_count, size_in_mb) for a model, assuming float32 params."""
    params = sum(p.numel() for p in model.parameters())
    return params, params * 4 / (1024 * 1024)
