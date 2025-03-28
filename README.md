---

### **📌 Credit Risk Modeling with Explainable AI**  

![Credit Risk](https://miro.medium.com/max/1400/1*fdu3Jb0J6xL2vPXh68-QwA.png)  

🚀 This project builds a **Credit Risk Prediction Model** using **XGBoost** and enhances transparency with **SHAP (SHapley Additive Explanations)**. The model predicts loan defaults using **Lending Club Loan Data** and provides feature explainability for better decision-making.  

---

## **📖 Table of Contents**  
- [📌 Credit Risk Modeling with Explainable AI](#-credit-risk-modeling-with-explainable-ai)  
- [📖 Table of Contents](#-table-of-contents)  
- [📂 Project Structure](#-project-structure)  
- [📊 Dataset](#-dataset)  
- [🛠️ Installation & Setup](#️-installation--setup)  
- [🚀 Model Training & Evaluation](#-model-training--evaluation)  
- [📝 Explainability with SHAP](#-explainability-with-shap)  
- [🌍 API Deployment (FastAPI)](#-api-deployment-fastapi)  
- [💡 Key Takeaways](#-key-takeaways)  
- [📌 Future Improvements](#-future-improvements)  

---

## **📂 Project Structure**  

```
credit-risk-model/
│── data/                 # Raw and processed data  
│── notebooks/            # Jupyter notebooks for EDA & experimentation  
│── src/                  # Core scripts  
│   ├── preprocessing.py   # Data preprocessing  
│   ├── train_model.py     # Model training (XGBoost)  
│   ├── explainability.py  # SHAP-based explainability  
│   ├── evaluate.py        # Model evaluation (Confusion Matrix, AUC-ROC)  
│   ├── deploy_api.py      # FastAPI deployment  
│── models/               # Saved models  
│── README.md             # Project documentation  
│── requirements.txt      # Dependencies  
│── .gitignore            # Ignore unnecessary files  
```

---

## **📊 Dataset**  
📌 We used **Lending Club Loan Data (2007-2018)**, consisting of:  
✅ **Accepted Loans** (`accepted_2007_to_2018Q4.csv`)  
✅ **Rejected Loans** (`rejected_2007_to_2018Q4.csv`)  

🔹 **Key Features:** Loan amount, Interest rate, Debt-to-Income ratio, Credit Score, Employment length, etc.  

---

## **🛠️ Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
```

### **2️⃣ Create a Virtual Environment**  
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Run the Model Training**  
```sh
python src/train_model.py
```

---

## **🚀 Model Training & Evaluation**  
✅ **Machine Learning Model:** `XGBoost`  
✅ **Metrics Evaluated:**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  
   - AUC-ROC Curve  

**Results:**  
```
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00    397995
           1       1.00      1.00      1.00     54146
```

---

## **📝 Explainability with SHAP**  
🔹 **Why Explainability?**  
Regulatory compliance requires transparency in credit scoring. SHAP values help explain model predictions.  

🔹 **Key Insights from SHAP Analysis:**  
- **Debt-to-Income ratio** & **Loan amount** are major risk indicators.  
- **Credit Score** has a strong impact on approval likelihood.  

Run SHAP analysis:  
```sh
python src/explainability.py
```
📊 **SHAP Summary Plot:**  
![SHAP Summary](https://shap.readthedocs.io/en/latest/_images/shap_summary_plot.png)

---

## **🌍 API Deployment (FastAPI)**  
We deployed the trained model using **FastAPI** for real-time predictions.  

✅ **Run API Server:**  
```sh
uvicorn src.deploy_api:app --reload
```

✅ **Test the API:**  
```sh
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"loan_amnt": 10000, "int_rate": 12.5, "dti": 15.3, "fico_score": 720}'
```

📌 API Response:  
```json
{"prediction": "Approved"}
```

---

## **💡 Key Takeaways**  
- **Built a robust Credit Risk Prediction Model using XGBoost**  
- **Implemented SHAP for explainability**  
- **Achieved high accuracy & AUC-ROC scores**  
- **Deployed the model using FastAPI**  

---

## **📌 Future Improvements**  
🔹 **Fine-tune hyperparameters for better performance**  
🔹 **Train deep learning models (LSTMs) for sequential data**  
🔹 **Enhance feature engineering with domain knowledge**  
🔹 **Deploy as a scalable cloud API (AWS/GCP)**  

---

### **🔗 Connect With Me**  
📩 **Email:** shreyasdole1105@gmail.com  
📂 **GitHub:** ShreyasDole
📄 **LinkedIn:** [your-linkedin-profile](https://www.linkedin.com/in/shreyas-dole/)  

🚀 **If you find this useful, don’t forget to ⭐ the repository!** 🚀  

---
