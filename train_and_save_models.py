import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

os.makedirs('models', exist_ok=True)

print("="*80)
print("  LOADING AND PREPARING DATA")
print("="*80)

file_path = "dataset/processed_data.csv"
df = pd.read_csv(file_path)
print(f"\n✓ Data loaded: {df.shape}")

X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Feature columns: {X.columns.tolist()}")
print(f"✓ Target distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Scaler fitted and applied")

joblib.dump(scaler, 'models/scaler.pkl')
print(f"✓ Scaler saved to 'models/scaler.pkl' (for reference)")

print("\n" + "="*80)
print("  TRAINING MODELS")
print("="*80)

models_to_train = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X_train, False),
    'AdaBoost': (AdaBoostClassifier(n_estimators=100, random_state=42), X_train, False),
    'SVM': (SVC(kernel='rbf', probability=True, random_state=42), X_train_scaled, True),
    'LightGBM': (LGBMClassifier(n_estimators=100, random_state=42, verbose=-1), X_train, False),
}

results = {}

for model_name, (model, X_train_data, use_scaled) in models_to_train.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train_data, y_train)

    X_test_data = X_test_scaled if use_scaled else X_test
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
    }
    
    results[model_name] = metrics

    joblib.dump(model, f'models/{model_name.lower().replace(" ", "_")}_model.pkl')
    
    print(f"  ✓ Trained and saved")
    print(f"    - Accuracy: {metrics['Accuracy']:.4f}")
    print(f"    - F1-Score: {metrics['F1-Score']:.4f}")
    print(f"    - ROC-AUC: {metrics['ROC-AUC']:.4f}")

results_df = pd.DataFrame(results).T
results_df.to_csv('models/model_metrics.csv')
print(f"\n✓ Model metrics saved to 'models/model_metrics.csv'")

print("\n" + "="*80)
print("  SUMMARY")
print("="*80)
print("\n" + results_df.to_string())

print("\n" + "="*80)
print("  ✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*80)
print("\nYou can now run the Streamlit app with:")
print("  streamlit run app.py")
