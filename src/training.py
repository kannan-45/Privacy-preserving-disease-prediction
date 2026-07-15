"""
Local model training with adaptive differential privacy applied to
gradients before each optimizer step.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train(model, X, y, epochs=100, lr=0.001, max_noise_scale=None):
    """Train a local model with class-balanced loss and adaptive DP noise.

    Adaptive DP: the noise scale added to gradients grows with the current
    training loss, giving more privacy protection early in training when
    the model (and its gradients) reveal the most about the local data.
    """
    ratio = (len(y) - y.sum()) / (y.sum() + 1e-5)
    pos_weight = torch.tensor(min(ratio.item(), 2.0))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()

        out = model(X)
        loss = criterion(out, y)
        loss.backward()

        noise_scale = 0.0001 * (1 + loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        for p in model.parameters():
            if p.grad is not None:
                p.grad += torch.normal(0, noise_scale, size=p.grad.shape)

        optimizer.step()

    return model
