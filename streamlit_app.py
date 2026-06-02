"""
Depression Risk Prediction Application
Data Science Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Depression Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional academic styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1f3a5f;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f3a5f;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f3a5f;
        padding-left: 0.8rem;
    }
    .result-box-high {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1.5rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .result-box-medium {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1.5rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .result-box-low {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1.5rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .disclaimer {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #495057;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL FILES
# ============================================================================
@st.cache_resource
def load_model_files():
    """Load the trained model and preprocessing objects."""
    try:
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        return model, scaler, le_dict
    except FileNotFoundError:
        st.error("Model files not found. Please ensure rf_model.pkl, scaler.pkl, and le_dict.pkl are in the application directory.")
        st.stop()

model, scaler, le_dict = load_model_files()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_input(user_data):
    """Preprocess user input for model prediction."""
    df = pd.DataFrame([user_data])

    for col in le_dict.keys():
        if col in df.columns:
            try:
                df[col] = le_dict[col].transform(df[col])
            except ValueError:
                st.error(f"Invalid value for {col}")
                return None

    df_scaled = scaler.transform(df)
    return df_scaled


def get_risk_category(probability):
    """Categorize risk level based on probability."""
    if probability >= 0.7:
        return "High Risk", "high"
    elif probability >= 0.4:
        return "Moderate Risk", "medium"
    else:
        return "Low Risk", "low"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<div class="main-header">Depression Risk Prediction System</div>', unsafe_allow_html=True)

st.markdown("""
This application predicts depression risk based on demographic, lifestyle, and health factors 
using a Random Forest classification model trained on 100,000 records.
""")

# Disclaimer
st.markdown("""
<div class="disclaimer">
<strong>Academic Disclaimer:</strong> This application is developed as part of a Data Science 
academic project for educational purposes. It is not a medical diagnostic tool and should not 
replace professional medical or psychological evaluation. If you are experiencing mental health 
concerns, please consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

# Sidebar - Model Information
st.sidebar.markdown("### Model Information")
st.sidebar.markdown("""
**Algorithm:** Random Forest Classifier  
**Training Data:** 100,000 records  
**Features:** 15 input variables  
**Test Accuracy:** 87.5%  
**Precision:** 85.2%  
**Recall:** 89.3%  
**ROC-AUC:** 0.923  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Comparison Model")
st.sidebar.markdown("""
**Logistic Regression**  
Accuracy: 84.5%  
ROC-AUC: 0.912  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Information")
st.sidebar.markdown("""
**Course:** Data Science  
**Type:** Final Project  
**Models Used:**
- Logistic Regression
- Random Forest Classifier
""")

# ============================================================================
# INPUT FORM
# ============================================================================
st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=80, value=40)
    num_children = st.slider("Number of Children", min_value=0, max_value=4, value=1)
    income = st.number_input("Annual Income (USD)", min_value=0, max_value=250000, value=50000, step=1000)
    marital_status = st.selectbox(
        "Marital Status",
        options=['Single', 'Married', 'Divorced', 'Widowed']
    )
    education = st.selectbox(
        "Education Level",
        options=['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD']
    )

with col2:
    employment = st.selectbox(
        "Employment Status",
        options=['Employed', 'Unemployed']
    )
    smoking = st.selectbox(
        "Smoking Status",
        options=['Non-smoker', 'Former', 'Current']
    )
    activity = st.selectbox(
        "Physical Activity Level",
        options=['Sedentary', 'Moderate', 'Active']
    )
    sleep = st.selectbox(
        "Sleep Quality",
        options=['Poor', 'Fair', 'Good']
    )
    diet = st.selectbox(
        "Dietary Habits",
        options=['Unhealthy', 'Moderate', 'Healthy']
    )

st.markdown('<div class="section-header">Health and Medical History</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    alcohol = st.selectbox(
        "Alcohol Consumption",
        options=['Low', 'Moderate', 'High']
    )
    mental_illness = st.selectbox(
        "History of Mental Illness",
        options=['No', 'Yes']
    )
    substance_abuse = st.selectbox(
        "History of Substance Abuse",
        options=['No', 'Yes']
    )

with col4:
    family_depression = st.selectbox(
        "Family History of Depression",
        options=['No', 'Yes']
    )
    chronic = st.selectbox(
        "Chronic Medical Conditions",
        options=['No', 'Yes']
    )

# ============================================================================
# PREDICTION
# ============================================================================
st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

if st.button("Run Risk Assessment", use_container_width=True):
    user_data = {
        'Age': age,
        'Number of Children': num_children,
        'Income': income,
        'Marital Status': marital_status,
        'Education Level': education,
        'Smoking Status': smoking,
        'Employment Status': employment,
        'Physical Activity Level': activity,
        'Sleep Patterns': sleep,
        'Dietary Habits': diet,
        'Alcohol Consumption': alcohol,
        'History of Mental Illness': mental_illness,
        'History of Substance Abuse': substance_abuse,
        'Family History of Depression': family_depression,
        'Chronic Medical Conditions': chronic
    }

    X_processed = preprocess_input(user_data)

    if X_processed is not None:
        prediction_proba = model.predict_proba(X_processed)[0]
        depression_probability = prediction_proba[1]
        risk_category, risk_level = get_risk_category(depression_probability)

        # Display metrics
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Risk Probability", f"{depression_probability*100:.1f}%")

        with col_b:
            st.metric("Risk Category", risk_category)

        with col_c:
            confidence = max(prediction_proba) * 100
            st.metric("Model Confidence", f"{confidence:.1f}%")

        st.markdown("---")

        # Display result box based on risk level
        if risk_level == "high":
            st.markdown(f"""
            <div class="result-box-high">
            <h4 style="margin-top:0;">Assessment Result: High Depression Risk</h4>
            <p>The model indicates a probability of <strong>{depression_probability*100:.1f}%</strong> 
            for depression risk based on the provided information. It is strongly recommended to 
            consult with a qualified mental health professional for a comprehensive evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == "medium":
            st.markdown(f"""
            <div class="result-box-medium">
            <h4 style="margin-top:0;">Assessment Result: Moderate Depression Risk</h4>
            <p>The model indicates a probability of <strong>{depression_probability*100:.1f}%</strong> 
            for depression risk based on the provided information. Consider lifestyle modifications 
            and consulting with a healthcare provider if symptoms persist.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-low">
            <h4 style="margin-top:0;">Assessment Result: Low Depression Risk</h4>
            <p>The model indicates a probability of <strong>{depression_probability*100:.1f}%</strong> 
            for depression risk based on the provided information. Continue maintaining healthy 
            lifestyle habits.</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors analysis
        st.markdown('<div class="section-header">Identified Risk Factors</div>', unsafe_allow_html=True)

        risk_factors = []
        protective_factors = []

        if user_data['History of Mental Illness'] == 'Yes':
            risk_factors.append("History of Mental Illness")
        if user_data['Family History of Depression'] == 'Yes':
            risk_factors.append("Family History of Depression")
        if user_data['History of Substance Abuse'] == 'Yes':
            risk_factors.append("History of Substance Abuse")
        if user_data['Employment Status'] == 'Unemployed':
            risk_factors.append("Unemployment")
        if user_data['Sleep Patterns'] == 'Poor':
            risk_factors.append("Poor Sleep Quality")
        if user_data['Physical Activity Level'] == 'Sedentary':
            risk_factors.append("Sedentary Lifestyle")
        if user_data['Alcohol Consumption'] == 'High':
            risk_factors.append("High Alcohol Consumption")
        if user_data['Chronic Medical Conditions'] == 'Yes':
            risk_factors.append("Chronic Medical Conditions")

        if user_data['Physical Activity Level'] == 'Active':
            protective_factors.append("Active Lifestyle")
        if user_data['Sleep Patterns'] == 'Good':
            protective_factors.append("Good Sleep Quality")
        if user_data['Dietary Habits'] == 'Healthy':
            protective_factors.append("Healthy Diet")
        if user_data['Employment Status'] == 'Employed':
            protective_factors.append("Employed")

        col_risk, col_protect = st.columns(2)

        with col_risk:
            st.markdown("**Risk Factors Present:**")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- No significant risk factors identified")

        with col_protect:
            st.markdown("**Protective Factors Present:**")
            if protective_factors:
                for factor in protective_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- Consider developing protective factors")

# ============================================================================
# MODEL PERFORMANCE SECTION
# ============================================================================
st.markdown("---")
st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

performance_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [0.8452, 0.8325, 0.8108, 0.8215, 0.9124],
    'Random Forest': [0.8750, 0.8520, 0.8930, 0.8720, 0.9230]
})

st.dataframe(performance_data, use_container_width=True, hide_index=True)

st.markdown("""
**Model Selection Rationale:** The Random Forest classifier was selected as the primary 
prediction model due to its superior performance across all evaluation metrics, particularly 
its higher recall (89.3%), which is critical in healthcare applications to minimize false 
negatives.
""")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem; padding: 1rem;'>
Depression Risk Prediction System | Data Science Final Project<br>
This application is for academic and educational purposes only.
</div>
""", unsafe_allow_html=True)
