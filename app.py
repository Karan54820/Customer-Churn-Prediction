import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        color: #d32f2f;
        font-weight: bold;
    }
    .low-risk {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression_model.pkl'),
        'Random Forest': joblib.load('models/random_forest_model.pkl'),
        'AdaBoost': joblib.load('models/adaboost_model.pkl'),
        'SVM': joblib.load('models/svm_model.pkl'),
        'LightGBM': joblib.load('models/lightgbm_model.pkl'),
    }
    return models


try:
    models = load_models()
except Exception as e:
    st.error(f"❌ Models not found. Please run 'train_and_save_models.py' first.\nError: {str(e)}")
    st.stop()


def get_predictions(features, selected_models):
    """Get predictions from selected models"""
    predictions = {}
    probabilities = {}
    
    for model_name in selected_models:
        model = models[model_name]
        
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0, 1]
        
        predictions[model_name] = pred
        probabilities[model_name] = proba
    
    return predictions, probabilities


st.title("🔮 Customer Churn Prediction")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Tenure")
    
    age = st.slider("Age", min_value=18, max_value=80, value=45)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=24)
    tenure_bucket = st.selectbox("Tenure Bucket", [0, 1, 2, 3])

with col2:
    st.subheader("Service Usage")
    
    usage_frequency = st.slider("Usage Frequency", min_value=0, max_value=30, value=15)
    support_calls = st.slider("Support Calls", min_value=0, max_value=10, value=3)
    payment_delay = st.slider("Payment Delay (days)", min_value=0, max_value=30, value=5)
    subscription_type = st.slider("Subscription Type", min_value=0, max_value=2, value=1)

with col3:
    st.subheader("Interactions & Spend")
    
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)
    last_interaction = st.slider("Last Interaction (days ago)", min_value=0, max_value=30, value=5)
    contract_duration = st.slider("Contract Duration (Months)", min_value=1, max_value=36, value=12)
    support_call_intensity = st.number_input("Support Call Intensity", min_value=0.0, max_value=1.0, value=0.1, step=0.01)


st.markdown("---")
st.subheader("Model Selection")
selected_models = st.multiselect(
    "Choose models for prediction:",
    list(models.keys()),
    default=["Random Forest", "LightGBM"]
)

if not selected_models:
    st.warning("Please select at least one model!")
    st.stop()

features_dict = {
    'Age': age,
    'Gender': gender,
    'Tenure': tenure,
    'Usage Frequency': usage_frequency,
    'Support Calls': support_calls,
    'Payment Delay': payment_delay,
    'Subscription Type': subscription_type,
    'Total Spend': total_spend,
    'Last Interaction': last_interaction,
    'Contract Duration (Months)': contract_duration,
    'Tenure_Bucket': tenure_bucket,
    'Support_Call_Intensity': support_call_intensity,
    'Interaction_Frequency': (usage_frequency / (tenure + 1)) if tenure > 0 else 0
}

if st.button("🔍 Predict Churn Risk", key="predict_button"):
    features_df = pd.DataFrame([features_dict])
    predictions, probabilities = get_predictions(features_df, selected_models)
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    cols = st.columns(len(selected_models))
    for idx, (model_name, col) in enumerate(zip(selected_models, cols)):
        with col:
            churn_status = "⚠️ CHURN" if predictions[model_name] == 1 else "✅ NO CHURN"
            prob = probabilities[model_name]
            
            st.metric(
                model_name,
                churn_status,
                f"Risk: {prob*100:.1f}%"
            )

            st.progress(prob, text=f"{prob*100:.1f}%")

    st.markdown("---")
    avg_churn_prob = np.mean(list(probabilities.values()))
    
    if avg_churn_prob > 0.7:
        st.error(f"**HIGH CHURN RISK** - Average Probability: {avg_churn_prob*100:.1f}%")
    elif avg_churn_prob > 0.4:
        st.warning(f"**MEDIUM CHURN RISK** - Average Probability: {avg_churn_prob*100:.1f}%")
    else:
        st.success(f" **LOW CHURN RISK** - Average Probability: {avg_churn_prob*100:.1f}%")