import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from preprocessing import load_accepted_data, split_data

def evaluate_model():
    """
    Evaluate the trained XGBoost model and generate performance metrics.
    """
    # Load model
    model = joblib.load("models/credit_risk_xgb.pkl")
    
    # Load and preprocess dataset
    df = load_accepted_data("data/accepted_2007_to_2018Q4.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["No Default", "Default"])
    plt.yticks([0, 1], ["No Default", "Default"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    print("Confusion matrix saved.")

    # AUC-ROC Curve
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve")
    plt.legend()
    plt.savefig("models/auc_roc_curve.png")
    print("AUC-ROC curve saved.")

if __name__ == "__main__":
    evaluate_model()
