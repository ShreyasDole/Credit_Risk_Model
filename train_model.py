import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from preprocessing import load_accepted_data, split_data

def train_xgboost_model():
    """
    Train an XGBoost model on accepted loans data.
    """
    # Load and split data
    df = load_accepted_data("data/accepted_2007_to_2018Q4.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions & evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/credit_risk_xgb.pkl")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_xgboost_model()
