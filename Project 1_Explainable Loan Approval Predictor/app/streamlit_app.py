"""
streamlit_app.py

Explainable Loan Approval Predictor Dashboard
----------------------------------------------
Run with:
    streamlit run app/streamlit_app.py

Requires the pipeline to have been run first so that the
following files exist in ./data/:
    model.pkl
    label_encoders.pkl
    feature_columns.pkl
    german_credit.csv
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Loan Approval Predictor',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: #0f1117;
    }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
    }

    .metric-card {
        background: linear-gradient(135deg, #1e2233, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .approved-banner {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }

    .denied-banner {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }

    .banner-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .banner-subtitle {
        font-size: 1rem;
        margin: 8px 0 0 0;
        opacity: 0.85;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 24px 0 12px 0;
        border-bottom: 1px solid #2d3250;
        padding-bottom: 8px;
    }

    .sidebar-info {
        background: #1e2233;
        border-radius: 12px;
        padding: 16px;
        font-size: 0.85rem;
        color: #8892b0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load model artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
    encoders = joblib.load(os.path.join(DATA_DIR, 'label_encoders.pkl'))
    features = joblib.load(os.path.join(DATA_DIR, 'feature_columns.pkl'))
    df_raw = pd.read_csv(os.path.join(DATA_DIR, 'german_credit.csv'))
    return model, encoders, features, df_raw


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def check_data_ready():
    required = ['model.pkl', 'label_encoders.pkl', 'feature_columns.pkl', 'german_credit.csv']
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    return missing


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Loan Approval Predictor")
st.markdown("Enter applicant details to receive a prediction and a full explanation of which factors drove the decision.")
st.markdown("---")

missing_files = check_data_ready()
if missing_files:
    st.error(f"Model files not found. Please run `python pipeline.py` first.\n\nMissing: {', '.join(missing_files)}")
    st.stop()

model, encoders, feature_cols, df_raw = load_model_artifacts()
explainer = get_explainer(model)


# ─── Sidebar: Applicant Input Form ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown('<div class="sidebar-info">Fill in the applicant information below. The model will predict approval probability and explain which factors mattered most.</div>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('<div class="section-header">Financial Profile</div>', unsafe_allow_html=True)

    checking_status_options = ['no checking', '0<=X<200', '<0', '>=200']
    checking_status = st.selectbox(
        'Checking account status',
        checking_status_options,
        help='Current balance in the applicant\'s checking account'
    )

    credit_amount = st.slider(
        'Loan amount (DM)', min_value=250, max_value=18500, value=3000, step=50
    )

    duration = st.slider(
        'Loan duration (months)', min_value=4, max_value=72, value=24, step=1
    )

    savings_status_options = ['no known savings', '<100', '100<=X<500', '500<=X<1000', '>=1000']
    savings_status = st.selectbox(
        'Savings account status',
        savings_status_options,
        help='Savings or bonds balance'
    )

    st.markdown('<div class="section-header">Credit History</div>', unsafe_allow_html=True)

    credit_history_options = [
        'no credits/all paid',
        'all paid',
        'existing paid',
        'delayed previously',
        'critical/other existing credit'
    ]
    credit_history = st.selectbox('Credit history', credit_history_options)

    purpose_options = [
        'new car', 'used car', 'furniture/equipment', 'radio/tv',
        'domestic appliance', 'repairs', 'education', 'vacation',
        'retraining', 'business', 'other'
    ]
    purpose = st.selectbox('Purpose of loan', purpose_options)

    st.markdown('<div class="section-header">Personal Profile</div>', unsafe_allow_html=True)

    age = st.slider('Age', min_value=19, max_value=75, value=35, step=1)

    employment_options = ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7']
    employment = st.selectbox('Employment (years)', employment_options)

    personal_status_options = [
        'male single',
        'female div/dep/mar',
        'male div/sep',
        'male mar/wid'
    ]
    personal_status = st.selectbox('Personal status', personal_status_options)

    housing_options = ['own', 'free', 'rent']
    housing = st.selectbox('Housing', housing_options)

    property_magnitude_options = ['real estate', 'life insurance', 'car', 'no known property']
    property_magnitude = st.selectbox('Property / collateral', property_magnitude_options)

    other_parties_options = ['none', 'co applicant', 'guarantor']
    other_parties = st.selectbox('Other parties (guarantors)', other_parties_options)

    num_dependents = st.slider('Number of dependents', 1, 2, 1)
    residence_since = st.slider('Residence duration (years)', 1, 4, 3)
    existing_credits = st.slider('Existing credits at this bank', 1, 4, 1)
    installment_commitment = st.slider('Installment rate (% of income)', 1, 4, 3)

    other_payment_plans_options = ['none', 'bank', 'stores']
    other_payment_plans = st.selectbox('Other payment plans', other_payment_plans_options)

    job_options = ['unskilled resident', 'unskilled non res', 'skilled', 'high qualif/self emp/mgmt']
    job = st.selectbox('Job type', job_options)

    own_telephone_options = ['none', 'yes']
    own_telephone = st.selectbox('Has telephone', own_telephone_options)

    foreign_worker_options = ['yes', 'no']
    foreign_worker = st.selectbox('Foreign worker', foreign_worker_options)

    predict_btn = st.button('Predict Approval', use_container_width=True, type='primary')


# ─── Build Input DataFrame ────────────────────────────────────────────────────
def encode_input(raw_input: dict, encoders: dict, feature_cols: list) -> pd.DataFrame:
    row = {}
    for col in feature_cols:
        val = raw_input.get(col)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            if val_str in le.classes_:
                row[col] = int(le.transform([val_str])[0])
            else:
                # Fallback to nearest class
                row[col] = 0
        else:
            row[col] = val
    return pd.DataFrame([row])[feature_cols]


raw_input = {
    'checking_status': checking_status,
    'duration': duration,
    'credit_history': credit_history,
    'purpose': purpose,
    'credit_amount': credit_amount,
    'savings_status': savings_status,
    'employment': employment,
    'installment_commitment': installment_commitment,
    'personal_status': personal_status,
    'other_parties': other_parties,
    'residence_since': residence_since,
    'property_magnitude': property_magnitude,
    'age': age,
    'other_payment_plans': other_payment_plans,
    'housing': housing,
    'existing_credits': existing_credits,
    'job': job,
    'num_dependents': num_dependents,
    'own_telephone': own_telephone,
    'foreign_worker': foreign_worker,
}

input_df = encode_input(raw_input, encoders, feature_cols)


# ─── Main panel: default view ─────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">1,000</div>
            <div class="metric-label">loan applicants</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">~77%</div>
            <div class="metric-label">accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Explainability</div>
            <div class="metric-value">SHAP</div>
            <div class="metric-label">per-decision explanation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How to use this tool")
    st.markdown("""
1. Fill in the applicant details in the sidebar on the left.
2. Click **Predict Approval**.
3. The tool will show you the prediction (Approved / Denied) with a confidence score.
4. A SHAP waterfall chart will explain exactly which factors drove the result and by how much.

**What is SHAP?** SHAP (SHapley Additive exPlanations) is a method from game theory that assigns each feature a contribution score for a given prediction. Red bars push toward approval. Blue bars push toward denial.
    """)

    shap_summary_path = os.path.join(DATA_DIR, 'shap_summary.png')
    if os.path.exists(shap_summary_path):
        st.markdown("### Global Feature Importance")
        st.markdown("This plot shows which features matter most across all applicants.")
        st.image(shap_summary_path, use_column_width=True)


# ─── Main panel: prediction result ───────────────────────────────────────────
if predict_btn:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    approved = prediction == 1

    # Result banner
    if approved:
        st.markdown(f"""
        <div class="approved-banner">
            <p class="banner-title" style="color:#6ee7b7;">✅ APPROVED</p>
            <p class="banner-subtitle" style="color:#a7f3d0;">Approval probability: {probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="denied-banner">
            <p class="banner-title" style="color:#fca5a5;">❌ DENIED</p>
            <p class="banner-subtitle" style="color:#fecaca;">Approval probability: {probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Decision</div>
            <div class="metric-value" style="color:{'#10b981' if approved else '#ef4444'}">
                {'Approved' if approved else 'Denied'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Approval Probability</div>
            <div class="metric-value">{probability:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Denial Probability</div>
            <div class="metric-value">{1 - probability:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # SHAP explanation
    st.markdown("### Why was this decision made?")
    st.markdown("The chart below shows each feature's contribution to this specific prediction. **Red bars** pushed the model toward *Approved*. **Blue bars** pushed toward *Denied*.")

    with st.spinner('Computing SHAP explanation...'):
        # SHAP 0.45+ returns an Explanation object with .values shape
        # (n_samples, n_features, n_classes) for binary classifiers
        shap_explanation = explainer(input_df)
        vals = shap_explanation.values

        if vals.ndim == 3:
            shap_vals_approved = vals[0, :, 1]
            base_value = float(shap_explanation.base_values[0, 1])
        else:
            shap_vals_approved = vals[0]
            base_value = float(shap_explanation.base_values[0])

        explanation = shap.Explanation(
            values=shap_vals_approved,
            base_values=base_value,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1a1d2e')
        ax.set_facecolor('#1a1d2e')

        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Feature value table
    st.markdown("---")
    st.markdown("### Applicant feature values used for this prediction")

    display_df = pd.DataFrame({
        'Feature': input_df.columns,
        'Value (encoded)': input_df.iloc[0].values,
        'SHAP Contribution': shap_vals_approved
    })
    display_df['SHAP Contribution'] = display_df['SHAP Contribution'].round(4)
    display_df = display_df.sort_values('SHAP Contribution', key=abs, ascending=False)
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(display_df, use_container_width=True)
