# Breast Cancer Dataset

This project uses the **Wisconsin Breast Cancer Diagnostic Dataset**, loaded directly via
`sklearn.datasets.load_breast_cancer()` — no manual download needed.

- 569 samples, 30 numeric features (mean/error/worst of radius, texture, perimeter, area, etc.)
- Binary target: malignant (0) / benign (1)
- Source: UCI Machine Learning Repository, bundled with scikit-learn

## Diabetes Dataset

The Pima Indians Diabetes dataset is fetched directly at runtime from:
`https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`

No manual download is required for either dataset — both are pulled automatically when you run `src/main.py`.
