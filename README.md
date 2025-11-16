
# Customer Churn Prediction

**Project:** Predict customer churn using preprocessing, feature engineering, and machine learning.

**Status:** Example preprocessing, exploratory analysis, and model training scripts included.

**Contents:** This repository contains notebooks and scripts to preprocess a customer churn dataset, explore features, train models, and save evaluation metrics.

**Prerequisites**
- **Python:** Recommended `3.8+`.
- **Dependencies:** Install from `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

**How to use**
- **Explore notebooks:** Open the notebooks (`.ipynb`) in the project root to reproduce EDA, preprocessing, and model experiments.
- **Run preprocessing / training scripts:**

```powershell
python train_and_save_models.py
python app.py
```

Use `train_and_save_models.py` to run model training and save models/metrics to the `models/` folder. Use `app.py` for quick local demonstration if applicable.

**Data**
- All datasets are in the `dataset/` folder.
- Key files:
  - `customer_churn_dataset-training-master.csv` — training raw data
  - `customer_churn_dataset-testing-master.csv` — testing raw data
  - `Preprocessed_data.csv`, `processed_data.csv`, `Feature_Engineered_data.csv` — transformed datasets used for modeling
  - `SMOTE_Balanced_data.csv` — balanced dataset for training experiments

**Repository layout**
- `app.py` : (optional) small demo / runner for prediction or inspection
- `train_and_save_models.py` : trains models and writes `models/` outputs
- `preprocessing_testing.py` : preprocessing utilities and tests
- `*.ipynb` : notebooks for EDA, preprocessing, feature engineering, and model training
- `dataset/` : CSV datasets and intermediate files
- `models/` : saved models and `model_metrics.csv`

**Quick commands**
- Run preprocessing and training (basic):

```powershell
python train_and_save_models.py
```

- Run a notebook (if you have Jupyter installed):

```powershell
jupyter notebook
```

**Notes & suggestions**
- Notebooks include exploratory plots and step-by-step preprocessing. Review `feature_engineering.ipynb` and `model_training.ipynb` for model details.
- If you want reproducible experiments, consider creating a virtual environment and pinning package versions.

**Outputs**
- Processed datasets: files in `dataset/` with `Preprocessed`, `Feature_Engineered`, or `SMOTE` in their names.
- Models & metrics: `models/model_metrics.csv` and any model artifacts saved by `train_and_save_models.py`.
