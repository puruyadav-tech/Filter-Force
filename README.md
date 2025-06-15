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

![image](https://github.com/user-attachments/assets/76357276-f33e-4ec6-abcf-42a88bea3ca9)

• pandas and numpy: Used for handling structured data and numerical operations.

• matplotlib.pyplot and seaborn: Used for plotting graphs and visualizing distributions and model results.

• TfidfVectorizer: Converts text into numerical features based on frequency and importance.

• RandomForestClassifier: (Though unused here) A machine learning model. You used XGBClassifier instead.

• XGBClassifier: An optimized version of gradient-boosted trees, very powerful for tabular data.

• train_test_split: Splits data into training and validation sets.

• classification_report, confusion_matrix, f1_score, etc.: Used to evaluate model performance.

• SMOTE: Balances the dataset by generating synthetic examples for the minority class (fraudulent jobs).

• shap: Helps explain the model’s predictions and shows which features are most important.

• warnings.filterwarnings("ignore"): Suppresses warnings for clean output.

![image](https://github.com/user-attachments/assets/bbaa63e7-e7f2-48cf-b056-39a4b1bdac1c)

• Reading CSV files into pandas DataFrames.

• on_bad_lines='skip' ensures any corrupted lines or format errors are skipped rather than crashing the program.

![image](https://github.com/user-attachments/assets/729bac06-9b45-4aa4-90cf-bc0602c47fcd)

🧹 Dataset Preparation & Feature Engineering
To ensure robust and consistent model training, we preprocess and enrich the dataset through a series of thoughtful transformations:

1. ✅ Safe Copy of the Data
We begin by copying the input DataFrame to avoid modifying the original data. This is a best practice for all preprocessing pipelines.

2. 🧾 Validate and Clean Text Columns
We ensure key textual columns (title, description, requirements, company_profile) exist. If missing, they're created and filled with empty strings to avoid errors during vectorization.

3. 📧 Clean the Email Column
If the email field exists, we replace all missing values with blank strings to maintain formatting and prevent errors during feature extraction.

4. 📚 Merge Text for TF-IDF
We combine all main text fields into a unified full_text column, which is used to generate TF-IDF features. This leverages the entire job listing content for better text-based learning.

5. 🧠 Generate Custom Features
We engineer several insightful features to detect fraud patterns:

desc_len: Length of the job description (in characters).

word_count: Number of words in the description.

num_digits_in_title: Count of numerical digits in the job title (e.g., “Earn $5000”).

has_profile: Boolean flag — does the company provide a profile?

suspicious_terms: Checks for keywords like "bitcoin", "transfer", "click", "money", etc., common in scams.

6. 📬 Email-Based Features
email_domain: Extracted from the email (e.g., gmail.com, company.com).

free_email: Flags whether the domain belongs to a free provider (like gmail, yahoo, etc.), often used in fake listings.

7. 🔄 Consistency Across Datasets
We apply the same transformation function to both training and test datasets. This guarantees consistent features during model training and inference.






