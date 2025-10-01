import os
import joblib
import shap

# Define model folder
MODEL_DIR = 'models'

try:
    prediction_model = joblib.load(os.path.join(MODEL_DIR, 'churn_model_xgb.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    anchor_date = joblib.load(os.path.join(MODEL_DIR, 'anchor_date.pkl'))
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessing artifacts: {e}")

try:
    explainer = shap.TreeExplainer(prediction_model)
except Exception as e:
    raise RuntimeError(f"Error initializing SHAP explainer: {e}")
