from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("C:\Credit Risk Using Advance ML\credit-risk-model\models\credit_risk_xgb.pkl")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running 🚀"}

@app.post("/predict/")
def predict(data: dict):
    """
    Predict loan approval risk based on input features.
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])

        # Ensure all features match training format
        expected_features = model.feature_names_in_
        df = df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()

        return {
            "prediction": int(prediction),
            "probability": probability
        }
    
    except Exception as e:
        return {"error": str(e)}

