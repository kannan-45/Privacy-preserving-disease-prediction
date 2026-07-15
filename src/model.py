"""
TabTransformer-based model architecture with a hybrid attention layer,
used as the local & global model in the federated learning pipeline.
"""

import torch
import torch.nn as nn


class TabTransformerModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.feature_embed = nn.Linear(input_size, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Hybrid attention layer (on top of the transformer encoder)
        self.attn = nn.Linear(64, 64)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # Step 1: Feature embedding
        x = self.feature_embed(x)          # (batch, 64)

        # Step 2: Convert to sequence
        x = x.unsqueeze(1)                 # (batch, 1, 64)

        # Step 3: Transformer encoder
        x = self.transformer(x)

        # Step 4: Remove sequence dim
        x = x.squeeze(1)                   # (batch, 64)

        # Step 5: Hybrid attention
        w = torch.softmax(self.attn(x), dim=1)
        x = x * w

        # Step 6: Output
        return self.fc(x)
