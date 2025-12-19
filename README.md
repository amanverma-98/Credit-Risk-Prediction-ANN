# 🏦 Credit Risk Prediction using ANN (PyTorch)
## 📌 Project Overview

This project builds an end-to-end Credit Risk Scoring system to predict whether a customer is likely to default on a loan.
The model is developed using a fully connected Artificial Neural Network (ANN) implemented from scratch in PyTorch, with Optuna-based hyperparameter tuning and early stopping.

The goal is to maximize ROC-AUC, which is the industry-standard metric for imbalanced financial datasets.

## 🎯 Problem Statement

Financial institutions need to assess the creditworthiness of customers to:

-Reduce loan default risk

-Improve decision-making

-Minimize financial losses

This is a binary classification problem:

-0 → No Default

-1 → Default

## 📊 Dataset

Source: Kaggle – Give Me Some Credit

Size: ~150,000 samples

Features: Demographic & financial attributes

Target: SeriousDlqin2yrs

### 🔗 Dataset Link:
https://www.kaggle.com/c/GiveMeSomeCredit

## 🧠 Model Architecture

The project uses a deep ANN with:

-Fully connected layers

-Batch Normalization

-ReLU activation

-Dropout regularization

-Output layer with logits (BCEWithLogitsLoss)

```
Input Features
   ↓
Linear → BatchNorm → ReLU → Dropout
   ↓
Linear → BatchNorm → ReLU → Dropout
   ↓
Linear → Output (Logits)
```

## ⚙️ Techniques Used
```
-Data Cleaning & Outlier Handling

-Feature Scaling (StandardScaler)

-Stratified Train–Validation Split

-Handling Class Imbalance using Pos Weight

-Batch Normalization

-Early Stopping

-Optuna Hyperparameter Tuning

-ROC-AUC Evaluation
```

## 📈 Evaluation Metrics
```
Since the dataset is highly imbalanced, accuracy alone is misleading.

Primary metrics used:

✅ ROC-AUC (Primary)

Recall (Default class)
```

## 🧪 Results
```
-Validation ROC-AUC: ~0.86+ (varies with tuning)

-Note: In credit risk modeling, ROC-AUC is preferred over accuracy as it evaluates ranking quality rather than raw classification rate.
```
## 🗂️ Project Structure
```
credit-risk-ann/
│── credit_risk_ann.ipynb
│
└── streamlit.py
│
├── requirements.txt
└── README.md
```
## 🚀 How to Run
### 1️⃣ Clone Repository
```
git clone https://github.com/amanvermaa/credit-risk-ann.git
cd credit-risk-ann
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Run Notebook

Open credit_risk_ann.ipynb in Jupyter / Colab and run all cells.

## 📦 Libraries Used

-Python

-PyTorch

-Scikit-learn

-Optuna

-Pandas, NumPy

## 💡 Key Learnings

-Why ROC-AUC is critical for imbalanced datasets

-How to build ANNs for tabular data

-Hyperparameter tuning using Optuna

-Importance of threshold tuning in business problems

-End-to-end ML workflow using PyTorch

## 🔥 Future Improvements

-Add SHAP explainability

-Compare ANN with XGBoost / LightGBM

-Cost-sensitive evaluation

-Deploy model using FastAPI

 
## 👨‍💻 Author

Aman Verma
Machine Learning & Deep Learning Enthusiast

🔗 GitHub: https://github.com/amanverma-98

🔗 LinkedIn: www.linkedin.com/in/aman-verma-8126b7324


### ⭐ If you find this useful

Give the repo a ⭐ and feel free to fork!
