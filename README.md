**Credit Risk Prediction using Machine Learning Pipelines **

This project builds an end-to-end supervised machine learning pipeline to predict whether a customer will default on a loan.
It follows data preprocessing, model building, evaluation, and hyperparameter tuning using scikit-learn.

The project is designed to apply and reinforce all major concepts learned in the DataCamp course: Supervised Learning with scikit-learn.

Business Problem: Financial institutions must assess credit risk before approving loans.
Objective: Predict whether a customer will default (1) or not default (0) on a loan based on demographic and financial information.

This is a binary classification problem with real-world business impact 

Dataset Source: Kaggle – Credit Risk Dataset


**Machine Learning Pipeline Architecture**
Raw Data
   ↓
ColumnTransformer
   ├── Numerical Pipeline
   │     ├── SimpleImputer (median)
   │     └── StandardScaler
   │
   └── Categorical Pipeline
         ├── SimpleImputer (most_frequent)
         └── OneHotEncoder
   ↓
Logistic Regression


Model Performance
Confusion Matrix

Strong performance on non-defaulters

Reasonable recall for defaulters

Key Metrics (Test Set)

Accuracy: ~87%

ROC-AUC: ~0.87

ROC Curve

The ROC curve demonstrates strong class separation, validating the model’s ability to distinguish defaulters from non-defaulters across thresholds.
