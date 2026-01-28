# Cardiovascular Disease Prediction

A machine learning project to predict cardiovascular disease using patient clinical data. The project uses 1000 patient records with 14 clinical features to build classification models with 90%+ accuracy.

## Project Structure

```
cardiovascular-disease-prediction/
├── data/
│   ├── Cardiovascular_Disease_Dataset_1.csv # Original dataset
│   └── *.csv, *.pkl                         # Processed data and models
├── notebooks/
│   ├── Data_Understanding.ipynb             # EDA and exploration
│   ├── Data-Preparation.ipynb               # Preprocessing and feature engineering  
│   ├── Modeling_Evaluation.ipynb            # Model training and evaluation
│   └── 6_Deployment.py                      # Streamlit app
├── requirements.txt
└── README.md
```

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis notebooks in order:**
   ```bash
   # Data exploration
   jupyter notebook notebooks/Data_Understanding.ipynb
   
   # Data preprocessing
   jupyter notebook notebooks/Data-Preparation.ipynb
   
   # Model training and evaluation
   jupyter notebook notebooks/Modeling_Evaluation.ipynb
   ```

3. **Deploy the model:**
   ```bash
   streamlit run notebooks/6_Deployment.py
   ```

## Key Results

- **Best Model:** Gradient Boosting (92% ROC AUC)
- **Dataset:** 1000 patients, 14 features, binary classification
- **Top Features:** Number of major vessels, chest pain type, exercise-induced angina
- **Pipeline:** Complete CRISP-DM methodology with feature engineering and hyperparameter tuning