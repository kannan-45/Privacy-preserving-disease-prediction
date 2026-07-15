"""
Privacy-Preserving Disease Prediction via Federated Learning
----------------------------------------------------------------
Orchestrates the full pipeline: data prep -> non-IID hospital split ->
local training with adaptive DP -> CKKS encryption -> trust-based
secure aggregation -> global model evaluation.
"""

import os
import torch

from data_preprocessing import prepare_data
from model import TabTransformerModel
from training import train
from encryption import create_context
from aggregation import TrustAggregator
from evaluation import evaluate, model_size

NUM_ROUNDS = 5
MODEL_SAVE_PATH = os.path.join("models", "global_model.pt")


def main():
    print("Loading and preprocessing data...")
    hospitals, test_sets, max_features = prepare_data()

    print("Setting up CKKS encryption context...")
    context = create_context()

    global_model = TabTransformerModel(max_features)
    aggregator = TrustAggregator()

    params, size_mb = model_size(global_model)
    print(f"Model Params: {params}, Size: {size_mb:.4f} MB")

    total_comm = 0

    for r in range(NUM_ROUNDS):
        print(f"\nRound {r + 1}")

        local_models, sizes, metrics = [], [], []

        for i, (X, y) in enumerate(hospitals):
            model = TabTransformerModel(max_features)
            model.load_state_dict(global_model.state_dict())

            model = train(model, X, y)
            m = evaluate(model, X, y)

            print(f"Hospital {i + 1}:", m)

            local_models.append(model)
            sizes.append(len(X))
            metrics.append(m)

        global_model = aggregator.aggregate(global_model, local_models, sizes, metrics, context)

        comm = 2 * len(hospitals) * size_mb
        total_comm += comm
        print(f"Communication this round: {comm:.4f} MB")

    print("\nFINAL RESULTS")
    X_test, y_test = test_sets["combined"]
    print("Combined:", evaluate(global_model, X_test, y_test))
    print("Diabetes:", evaluate(global_model, *test_sets["diabetes"]))
    print("Cancer:", evaluate(global_model, *test_sets["cancer"]))

    print(f"\nTotal Communication Cost: {total_comm:.4f} MB")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(global_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nGlobal model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
