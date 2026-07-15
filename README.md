# 🔐 Privacy-Preserving Disease Prediction using Federated Learning

## Overview

This project implements a secure federated learning framework for disease prediction using **TabTransformer** models, **Adaptive Differential Privacy**, **Homomorphic Encryption (CKKS)**, and **Trust-Based Secure Aggregation**.

The system allows multiple hospitals to collaboratively train a disease prediction model **without ever sharing raw patient data**.

---

## Motivation

Healthcare organizations often cannot share patient records because of privacy regulations (e.g. HIPAA, GDPR).

Federated Learning enables collaborative model training while keeping data inside each hospital. To further strengthen privacy, this project integrates:

- Adaptive Differential Privacy
- Homomorphic Encryption
- Trust-Based Secure Aggregation

---

## Folder Structure

```
Privacy-Preserving-Disease-Prediction/
│
├── data/
│   └── breast_cancer_dataset_info.md   # dataset notes (both datasets are fetched at runtime)
│
├── models/
│   └── global_model.pt                 # saved after training (git-ignored)
│
├── results/
│   ├── communication_cost.png
│   ├── performance_comparison.png
│   ├── architecture.png
│   └── sample_output.txt
│
├── src/
│   ├── data_preprocessing.py           # loading, scaling, non-IID hospital split
│   ├── model.py                        # TabTransformerModel
│   ├── training.py                     # local training with adaptive DP
│   ├── encryption.py                   # CKKS encryption / decryption
│   ├── aggregation.py                  # trust-based secure aggregation
│   ├── evaluation.py                   # metrics + communication cost
│   └── main.py                         # orchestrates the full pipeline
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Technologies

- Python
- PyTorch
- TenSEAL (CKKS homomorphic encryption)
- NumPy / Pandas
- Scikit-learn

---

## Datasets

| Dataset | Source | Access |
|---|---|---|
| **Diabetes** | Pima Indians Diabetes Dataset | fetched at runtime from a public CSV URL |
| **Breast Cancer** | Wisconsin Breast Cancer Diagnostic Dataset | loaded via `sklearn.datasets.load_breast_cancer()` |

No manual dataset download is required — both are pulled automatically when you run `src/main.py`.

---

## Key Features

- Federated Learning across simulated hospitals
- Non-IID data distribution per hospital
- TabTransformer architecture with hybrid attention
- Adaptive Differential Privacy (noise scaled to training loss)
- CKKS Homomorphic Encryption of model updates
- Trust-Based secure aggregation (weighted by Precision/Recall/F1 + consistency)
- Communication cost tracking across federated rounds

---

## System Workflow

1. Load diabetes & breast cancer datasets
2. Preprocess (scale features, stratified train/test split, pad to equal width)
3. Split each dataset into non-IID hospital partitions
4. Train a local model per hospital with adaptive DP-protected gradients
5. Encrypt each local model's parameters with CKKS
6. Aggregate encrypted updates using trust-based weighting
7. Decrypt and update the global model
8. Evaluate on held-out test data
9. Repeat for multiple federated rounds, then save the global model

---

## Model Architecture

- Feature Embedding (Linear layer)
- Transformer Encoder (2 layers, 4 heads)
- Hybrid Attention layer
- Fully Connected head (64 → 32 → 1)
- Sigmoid output (via `BCEWithLogitsLoss` during training)

---

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## Security Techniques

### Differential Privacy
Gaussian noise, scaled adaptively to the current training loss, is added to gradients before each optimizer step to reduce information leakage from any single hospital's data.

### Homomorphic Encryption
Model parameters are encrypted using the CKKS scheme (via TenSEAL) so that weighted aggregation across hospitals happens **without ever decrypting individual updates**.

### Trust-Based Aggregation
Each hospital receives a trust score derived from its Precision, Recall, F1 score, and consistency with its own performance in the previous round — reducing the influence of unreliable or noisy clients.

---

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

This will download both datasets, run 5 rounds of federated training across 6 simulated hospitals, print per-hospital and global metrics each round, and save the trained global model to `models/global_model.pt`.

---

## Results

The proposed framework demonstrates secure collaborative learning while maintaining strong predictive performance on both the diabetes and breast cancer datasets. See the `results/` folder for architecture diagrams, communication cost plots, and sample output logs.

---

## Future Work

- Flower framework integration
- Real hospital deployment
- Blockchain-based client verification
- Secure Multi-Party Computation (SMPC)
- Explainable AI (XAI) for model interpretability

---

## Author

**Kannan Mandalreddy**
📧 kannan028864@gmail.com
