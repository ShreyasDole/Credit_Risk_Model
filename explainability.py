import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_accepted_data, split_data

def explain_model():
    """
    Explain the trained XGBoost model using SHAP values.
    """
    # Load model
    model = joblib.load("models/credit_risk_xgb.pkl")
    
    # Load and preprocess dataset
    df = load_accepted_data("data/accepted_2007_to_2018Q4.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Initialize SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Feature importance summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test)
    plt.savefig("models/shap_summary_plot.png")
    print("SHAP summary plot saved.")

    # Explain a single prediction
    sample_idx = 0  # Choose a sample from test set
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(shap_values[sample_idx])
    plt.savefig(f"models/shap_waterfall_{sample_idx}.png")
    print(f"SHAP waterfall plot for sample {sample_idx} saved.")

if __name__ == "__main__":
    explain_model()
