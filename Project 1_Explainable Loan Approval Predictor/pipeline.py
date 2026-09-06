"""
pipeline.py

Explainable Loan Approval Predictor
------------------------------------
This script runs the full ML pipeline end to end:
  1. Load the German Credit dataset from OpenML
  2. Encode categorical features
  3. Train a Random Forest classifier
  4. Evaluate the model on the test set
  5. Generate and save SHAP explanations
  6. Save the model and encoders to disk

Run with:
    python pipeline.py

Output files written to ./data/:
    model.pkl               trained Random Forest
    label_encoders.pkl      fitted LabelEncoders per categorical column
    feature_columns.pkl     ordered list of feature names
    german_credit.csv       raw dataset
    german_credit_encoded.csv  encoded dataset
    shap_summary.png        global feature importance plot
    shap_waterfall.png      local explanation for one sample
"""

import os
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5


# ─── Step 1: Load Data ────────────────────────────────────────────────────────
def load_data():
    print('[1/5] Loading German Credit dataset from OpenML...')
    credit = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
    df = credit.frame.copy()
    df['target'] = (df['class'] == 'good').astype(int)
    df.drop(columns=['class'], inplace=True)
    df.to_csv(os.path.join(DATA_DIR, 'german_credit.csv'), index=False)
    print(f'    Loaded {len(df)} rows, {df.shape[1] - 1} features.')
    return df


# ─── Step 2: Encode Features ─────────────────────────────────────────────────
def encode_features(df):
    print('[2/5] Encoding categorical features...')
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include=['object', 'category']).columns.tolist()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    df_enc.to_csv(os.path.join(DATA_DIR, 'german_credit_encoded.csv'), index=False)

    X = df_enc.drop(columns=['target'])
    y = df_enc['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    joblib.dump(encoders, os.path.join(DATA_DIR, 'label_encoders.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(DATA_DIR, 'feature_columns.pkl'))

    print(f'    {len(cat_cols)} categorical columns encoded.')
    print(f'    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples')
    return X_train, X_test, y_train, y_test, encoders


# ─── Step 3: Train Model ──────────────────────────────────────────────────────
def train_model(X_train, y_train):
    print('[3/5] Training Random Forest classifier...')
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(DATA_DIR, 'model.pkl'))
    print('    Model trained and saved to data/model.pkl')
    return model


# ─── Step 4: Evaluate Model ───────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    print('[4/5] Evaluating model...')
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f'    Accuracy : {acc:.4f}')
    print(f'    ROC-AUC  : {roc:.4f}')
    print()
    print(classification_report(y_test, y_pred, target_names=['Denied', 'Approved']))

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('    Confusion matrix saved to data/confusion_matrix.png')


# ─── Step 5: SHAP Explanations ────────────────────────────────────────────────
def explain_with_shap(model, X_test):
    print('[5/5] Computing SHAP values...')
    explainer = shap.TreeExplainer(model)

    # SHAP 0.45+ returns an Explanation object when called as a function.
    # Its .values shape is (n_samples, n_features, n_classes) for binary classifiers.
    shap_explanation = explainer(X_test)
    vals = shap_explanation.values

    if vals.ndim == 3:
        # New API: shape (n_samples, n_features, n_classes) — select class 1 (Approved)
        shap_vals_approved = vals[:, :, 1]
        base_value = float(shap_explanation.base_values[0, 1])
    else:
        # Older API: shape (n_samples, n_features)
        shap_vals_approved = vals
        base_value = float(shap_explanation.base_values[0])

    # Global summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals_approved, X_test, show=False)
    plt.title('SHAP Summary: Feature Impact on Loan Approval', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('    SHAP summary plot saved to data/shap_summary.png')

    # Local waterfall for first test sample
    sample_idx = 0
    explanation = shap.Explanation(
        values=shap_vals_approved[sample_idx],
        base_values=base_value,
        data=X_test.iloc[sample_idx].values,
        feature_names=X_test.columns.tolist()
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'shap_waterfall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('    SHAP waterfall plot saved to data/shap_waterfall.png')


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('  Explainable Loan Approval Predictor  Pipeline')
    print('=' * 55)

    df = load_data()
    X_train, X_test, y_train, y_test, encoders = encode_features(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    explain_with_shap(model, X_test)

    print()
    print('Pipeline complete. All outputs saved to ./data/')
