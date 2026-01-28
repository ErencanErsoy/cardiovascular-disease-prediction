#!/usr/bin/env python3
"""
6. CRISP-DM: Deployment - Cardiovascular Disease Prediction App

This Streamlit application provides a user-friendly interface for predicting
cardiovascular disease risk based on patient data. It uses the best trained model
and applies the same preprocessing pipeline as used during training.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #000000;
        color: #ffffff;
    }
    .high-risk {
        border-left: 5px solid #f44336;
    }
    .low-risk {
        border-left: 5px solid #4caf50;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_model_and_preprocessing():
    """Load the best trained model and preprocessing objects"""
    try:
        # Try different possible paths for the models directory
        possible_model_paths = ['../models', '../../models', 'models', '../cardiovascular-disease-prediction/models']
        model_dir = None
        
        for path in possible_model_paths:
            if os.path.exists(path):
                model_dir = path
                break
        
        if model_dir is None:
            st.error("❌ Models directory not found. Please run the modeling notebook first to train and save models.")
            return None, None, None
        
        # Try to find the best model file
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
        if model_files:
            model_path = os.path.join(model_dir, model_files[0])
            model = joblib.load(model_path)
            st.success(f"✅ Loaded model: {model_files[0]} from {model_dir}")
        else:
            st.error("❌ No trained model found. Please run the modeling notebook first.")
            return None, None, None
        
        # Try different possible paths for the data directory
        possible_data_paths = ['../data', '../../data', 'data', '../cardiovascular-disease-prediction/data']
        data_dir = None
        
        for path in possible_data_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'scaler.pkl')):
                data_dir = path
                break
        
        if data_dir is None:
            st.error("❌ Data directory or preprocessing files not found. Please run the data preparation notebook first.")
            return None, None, None
        
        # Load scaler and feature names
        scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
        feature_names = joblib.load(os.path.join(data_dir, 'selected_features.pkl'))
        
        st.success(f"✅ Loaded preprocessing files from {data_dir}")
        
        return model, scaler, feature_names
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please ensure you have run both the Data Preparation and Modeling notebooks first.")
        return None, None, None

def preprocess_input(patient_data, scaler, feature_names):
    """Preprocess patient input to match training data format"""
    
    # Create a DataFrame with the patient data
    df = pd.DataFrame([patient_data])
    
    # Feature engineering (same as in data preparation)
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '60+'])
    
    # Create cholesterol risk categories
    df['cholesterol_risk'] = pd.cut(df['serumcholestrol'], 
                                   bins=[0, 200, 240, np.inf], 
                                   labels=['Normal', 'Borderline', 'High'])
    
    # Create blood pressure categories
    df['bp_category'] = pd.cut(df['restingBP'], 
                              bins=[0, 120, 140, np.inf], 
                              labels=['Normal', 'Elevated', 'High'])
    
    # Create heart rate categories
    df['hr_category'] = pd.cut(df['maxheartrate'], 
                              bins=[0, 100, 150, np.inf], 
                              labels=['Low', 'Normal', 'High'])
    
    # One-hot encode categorical features
    categorical_features = ['age_group', 'cholesterol_risk', 'bp_category', 'hr_category']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Ensure all required columns are present
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only the features used in training
    df_final = df_encoded[feature_names]
    
    # Scale numerical features
    numerical_features = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']
    df_scaled = df_final.copy()
    
    # Only scale if these features exist in the final dataset
    existing_numerical = [col for col in numerical_features if col in df_scaled.columns]
    if existing_numerical:
        df_scaled[existing_numerical] = scaler.transform(df_scaled[existing_numerical])
    
    return df_scaled

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Cardiovascular Disease Risk Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Cardiovascular Disease Prediction Tool
    This application uses machine learning to assess cardiovascular disease risk based on patient data. 
    Please enter the patient information below to get a risk assessment.
    """)
    
    # Debug information (can be commented out in production)
    with st.expander("🔧 Debug Information", expanded=False):
        st.write(f"**Current working directory:** {os.getcwd()}")
        st.write("**Available directories:**")
        for item in ['../models', '../data', 'models', 'data']:
            exists = os.path.exists(item)
            st.write(f"- {item}: {'✅ Exists' if exists else '❌ Not found'}")
    
    # Load model and preprocessing objects
    model, scaler, feature_names = load_model_and_preprocessing()
    
    if model is None:
        st.stop()
    
    # Sidebar for patient information
    st.sidebar.header("👤 Patient Information")
    
    with st.sidebar:
        st.subheader("Basic Information")
        
        # Patient ID (for reference only)
        patient_id = st.text_input("Patient ID (optional)", value="", placeholder="Enter patient ID")
        
        # Age
        age = st.slider("Age", min_value=20, max_value=100, value=50, 
                       help="Patient's age in years")
        
        # Gender
        gender = st.selectbox("Gender", options=[1, 0], 
                             format_func=lambda x: "Male" if x == 1 else "Female",
                             help="Patient's biological sex")
        
        st.subheader("Clinical Measurements")
        
        # Chest Pain Type
        chestpain = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                format_func=lambda x: {
                                    0: "Typical Angina",
                                    1: "Atypical Angina", 
                                    2: "Non-anginal Pain",
                                    3: "Asymptomatic"
                                }[x],
                                help="Type of chest pain experienced")
        
        # Resting Blood Pressure
        restingBP = st.slider("Resting Blood Pressure (mmHg)", 
                             min_value=80, max_value=200, value=120,
                             help="Resting blood pressure in mmHg")
        
        # Serum Cholesterol
        serumcholestrol = st.slider("Serum Cholesterol (mg/dl)", 
                                   min_value=100, max_value=400, value=200,
                                   help="Serum cholesterol level in mg/dl")
        
        # Fasting Blood Sugar
        fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                        options=[0, 1],
                                        format_func=lambda x: "Yes" if x == 1 else "No",
                                        help="Whether fasting blood sugar > 120 mg/dl")
        
        # Resting Electrocardiographic Results
        restingrelectro = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                      format_func=lambda x: {
                                          0: "Normal",
                                          1: "ST-T Wave Abnormality",
                                          2: "Left Ventricular Hypertrophy"
                                      }[x],
                                      help="Resting electrocardiographic results")
        
        # Maximum Heart Rate
        maxheartrate = st.slider("Maximum Heart Rate Achieved", 
                                min_value=60, max_value=220, value=150,
                                help="Maximum heart rate achieved during exercise")
        
        # Exercise Induced Angina
        exerciseangia = st.selectbox("Exercise Induced Angina", 
                                    options=[0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No",
                                    help="Exercise induced angina")
        
        # ST Depression (Oldpeak)
        oldpeak = st.slider("ST Depression (Oldpeak)", 
                           min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                           help="ST depression induced by exercise relative to rest")
        
        # Slope of Peak Exercise ST Segment
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3],
                            format_func=lambda x: {
                                1: "Upsloping",
                                2: "Flat", 
                                3: "Downsloping"
                            }[x],
                            help="Slope of the peak exercise ST segment")
        
        # Number of Major Vessels
        noofmajorvessels = st.selectbox("Number of Major Vessels", 
                                       options=[0, 1, 2, 3],
                                       help="Number of major vessels colored by fluoroscopy")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Patient Data Summary")
        
        # Create patient data dictionary
        patient_data = {
            'patientid': patient_id if patient_id else 0,
            'age': age,
            'gender': gender,
            'chestpain': chestpain,
            'restingBP': restingBP,
            'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar,
            'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate,
            'exerciseangia': exerciseangia,
            'oldpeak': oldpeak,
            'slope': slope,
            'noofmajorvessels': noofmajorvessels
        }
        
        # Display patient data in a nice format
        data_display = {
            "Age": f"{age} years",
            "Gender": "Male" if gender == 1 else "Female",
            "Chest Pain": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[chestpain],
            "Resting BP": f"{restingBP} mmHg",
            "Cholesterol": f"{serumcholestrol} mg/dl",
            "Fasting Blood Sugar": "High (>120 mg/dl)" if fastingbloodsugar == 1 else "Normal (≤120 mg/dl)",
            "Resting ECG": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[restingrelectro],
            "Max Heart Rate": f"{maxheartrate} bpm",
            "Exercise Angina": "Yes" if exerciseangia == 1 else "No",
            "ST Depression": f"{oldpeak}",
            "ST Slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[slope],
            "Major Vessels": f"{noofmajorvessels}"
        }
        
        # Create a nice table
        df_display = pd.DataFrame(list(data_display.items()), columns=['Parameter', 'Value'])
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎯 Risk Assessment")
        
        if st.button("🔍 Predict Cardiovascular Disease Risk", type="primary", use_container_width=True):
            try:
                # Preprocess the input
                processed_data = preprocess_input(patient_data, scaler, feature_names)
                
                # Make prediction
                prediction = model.predict(processed_data)[0]
                prediction_proba = model.predict_proba(processed_data)[0]
                
                # Display results
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h3>⚠️ HIGH RISK</h3>
                        <p><strong>The patient has a high risk of cardiovascular disease.</strong></p>
                        <p><em>Recommendation: Immediate medical consultation advised</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h3>✅ LOW RISK</h3>
                        <p><strong>The patient has a low risk of cardiovascular disease.</strong></p>
                        <p><em>Recommendation: Continue regular health monitoring</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk factors analysis
                st.subheader("🔍 Key Risk Factors")
                
                risk_factors = []
                if restingBP > 140:
                    risk_factors.append("High Blood Pressure")
                if serumcholestrol > 240:
                    risk_factors.append("High Cholesterol")
                if age > 60:
                    risk_factors.append("Advanced Age")
                if exerciseangia == 1:
                    risk_factors.append("Exercise-Induced Angina")
                if maxheartrate < 100:
                    risk_factors.append("Low Maximum Heart Rate")
                
                if risk_factors:
                    st.warning("⚠️ **Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ No major risk factors identified")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
    Always consult with a healthcare professional for proper medical diagnosis and treatment.</p>
    <p>Developed using machine learning models trained on cardiovascular disease data.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()