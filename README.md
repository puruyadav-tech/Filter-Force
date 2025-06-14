# fraud-detector-app
🔍 Job Fraud Detector | Anveshan Hackathon Project An ML-powered Streamlit web app that detects fake job listings using natural language processing, feature engineering, and XGBoost. Includes real-time predictions, SHAP explanations, search, filtering, and CSV export — built to protect job seekers from online scams
https://drive.google.com/file/d/1oxygynnHOAU3NU3S8xHRdBS9WA_K1E7y/view?usp=sharing

 Fraud Detector: Smart Job Scam Detection System
🧠 Project Overview
Online job platforms are frequently targeted by scammers. These fake job listings waste applicants’ time and put their personal information at risk.

This project builds a machine learning pipeline that automatically detects fraudulent job listings and presents insights in a Streamlit-powered interactive dashboard.

🎯 Problem Statement
Develop a system to classify job listings as fraudulent or genuine using structured and unstructured data. Display predictions and insights in an intuitive dashboard.

🚀 Automatic Job Fraud Detection
Classifies job listings as real or fake using advanced machine learning techniques.

🧠 Hybrid Feature Engineering
Combines TF-IDF text vectors with structured numeric features:

-Description length & word count

-Suspicious keywords (e.g., money, bitcoin)

-Free email domain indicators (e.g., gmail.com, yahoo.com)

-Digit count in job titles

🌲 XGBoost Classifier for Accuracy & Speed

Leverages the power of XGBoost — a high-performance gradient boosting algorithm — for fast training and accurate predictions.

⚖️ Class Imbalance Handling with SMOTE
Uses SMOTE oversampling to improve detection of rare fraudulent cases without overfitting.

🧪 F1-Optimized Threshold Tuning

Dynamically selects the best threshold using precision-recall curve to maximize the F1-score — ideal for imbalanced data.

📊 Visual Analytics Dashboard

-Histogram of fraud probabilities

-Pie chart of predicted real vs. fake listings

-Threshold vs. F1-score visualization

-SHAP summary plots for feature importance

🔍 Explainable AI with SHAP

-Visualize top contributing features

-Interpret individual predictions

-Enhance transparency and trust in model decisions

✉️ High-Risk Job Alert System

Automatically flags listings with fraud probability ≥ 0.8 and supports email alerts via SMTP integration.

🔗 Unified Text + Numeric Feature Pipeline

Combines TF-IDF matrix with custom features using scipy.sparse.hstack for seamless model input.

⚡ Deployment-Ready with Streamlit

Integrated with a user-friendly Streamlit interface for real-time fraud prediction and insights.

🛠️ Technologies Used


| Area          | Tools/Libraries                      |
| ------------- | ------------------------------------ |
| Data Handling | `pandas`, `numpy`                    |
| Modeling      | `XGBoost`, `SMOTE`                   |
| NLP           | `TfidfVectorizer`                    |
| Evaluation    | `F1-score`, `precision-recall` curve |
| Visualization | `matplotlib`, `seaborn`, `SHAP`      |
| Web Dashboard | `Streamlit`                          |
| Deployment    | `Streamlit Cloud`                    |


🧾 Setup Instructions
