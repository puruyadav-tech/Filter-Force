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

 Dataset Preparation & Feature Engineering
To ensure robust and consistent model training, we preprocess and enrich the dataset through a series of thoughtful transformations:

1. Safe Copy of the Data
We begin by copying the input DataFrame to avoid modifying the original data. This is a best practice for all preprocessing pipelines.

2. Validate and Clean Text Columns
We ensure key textual columns (title, description, requirements, company_profile) exist. If missing, they're created and filled with empty strings to avoid errors during vectorization.

3. Clean the Email Column
If the email field exists, we replace all missing values with blank strings to maintain formatting and prevent errors during feature extraction.

4. Merge Text for TF-IDF
We combine all main text fields into a unified full_text column, which is used to generate TF-IDF features. This leverages the entire job listing content for better text-based learning.

5. Generate Custom Features
• We engineer several insightful features to detect fraud patterns:

• desc_len: Length of the job description (in characters).

• word_count: Number of words in the description.

• num_digits_in_title: Count of numerical digits in the job title (e.g., “Earn $5000”).

• has_profile: Boolean flag — does the company provide a profile?

• suspicious_terms: Checks for keywords like "bitcoin", "transfer", "click", "money", etc., common in scams.

6. Email-Based Features
email_domain: Extracted from the email (e.g., gmail.com, company.com).

free_email: Flags whether the domain belongs to a free provider (like gmail, yahoo, etc.), often used in fake listings.

7.  Consistency Across Datasets
We apply the same transformation function to both training and test datasets. This guarantees consistent features during model training and inference.


![image](https://github.com/user-attachments/assets/a56f1f07-2b84-4faa-984f-5c485883da3a)

1. Split the dataset:

• X: Input features from the training data.

• y: Output labels (0 = Real, 1 = Fraudulent).

• X_test: Input features from the test data (labels are not available).

2. Features in X:

• Text feature: Combined text column (formed by concatenating title, description, requirements, company_profile).

• Handcrafted numeric features:

• desc_len: Number of characters in the job description.

• word_count: Number of words in the description.

• num_digits_in_title: Number of digits in the job title.

• has_profile: Binary flag indicating presence of a company profile.

• suspicious_terms: Binary flag if scammy terms like money, bitcoin exist.

• free_email: Binary flag for free email domains (e.g., gmail.com).

3.  TF-IDF Vectorization

• To handle text data, we use TF-IDF (Term Frequency-Inverse Document Frequency) — a statistical method to evaluate how important a word is in a document relative to the entire corpus 

4. Configuration:

• max_features = 5000: Use the 5000 most informative words.

• stop_words = 'english': Removes common unimportant words (e.g., the, and, is).

5. Apply TF-IDF:

• On X['text'] → creates X_tfidf (sparse matrix).

• On X_test['text'] → creates X_test_tfidf (using the same vocabulary as training).

6. Combining All Features

The TF-IDF matrix (text features),

With 6 handcrafted numerical features (from feature engineering).

We use scipy.sparse.hstack() to horizontally stack these two sets of features into one big feature matri.

7. Final matrices:

• X_combined: Final feature set for training.

• X_test_combined: Final feature set for test predictions.

This combined matrix is now ready to feed into machine learning models like XGBoost  giving the model both:
These combined matrices are then used to train ML models like XGBoost or Random Forest, benefiting from:

• Rich semantic content (from text)

• Strong rule-based signals (from numeric features

![image](https://github.com/user-attachments/assets/1ecbd0f9-25ef-4c36-b796-01609c9bc1d2)

Handling Class Imbalance and Model Training

1. Problem: Class Imbalance

     • Fraudulent job postings are rare.

     • This causes class imbalance — a major challenge in machine learning.

     • A model that always predicts “not fraud” may achieve high accuracy but is useless.

2. Solution: SMOTE (Synthetic Minority Oversampling Technique)

 • SMOTE creates synthetic fraud examples by interpolating between real ones.

 • This balances the dataset and helps the model learn fraud patterns more effectively.

3. Train-Validation Split

 • Dataset is split using train_test_split() with stratify=y to maintain class distribution.

4. Model: XGBoost Classifier

 • Chosen for its speed and accuracy in classification tasks.

 • Key hyperparameters:

     • n_estimators=200: Number of trees.

     • max_depth=6: Controls model complexity to avoid overfitting.

     • subsample & colsample_bytree: Randomly sample rows and features to enhance generalization.

     • eval_metric='logloss': Metric to evaluate binary classification error.

5. Training

     • The model is trained on SMOTE-balanced data: X_res, y_res.

 ![image](https://github.com/user-attachments/assets/f9454534-08d6-4d88-82a0-f31798dac96c)

• Threshold Optimization for Better Performance

1. Default Threshold Issue

  • Most models use a default threshold of 0.5 for binary classification.

  • On imbalanced datasets, this may not yield optimal results.

2. Threshold Tuning

   • Use precision_recall_curve to compute scores across all possible thresholds.

   • Calculate F1 Score (harmonic mean of precision and recall) for each threshold.

   • Select the best_threshold that maximizes F1 Score.

3. Plot F1 vs Threshold

• Helps visualize how F1 score changes with different classification thresholds.

• Aids in selecting the optimal best_threshold.

4. Final Predictions with Best Threshold

• Apply best_threshold to the model's output probabilities.

• Generate predictions for evaluation.

5. Evaluate with classification_report()

• Provides detailed performance metrics:

    • Precision: Of all predicted frauds, how many were actually fraud?

    • Recall: Of all actual frauds, how many did the model detect?

    • F1 Score: Harmonic mean of precision and recall — balances false positives and false negatives.

    • Support: Number of true instances in each class.

6. Evaluate with confusion_matrix()

  • Breaks down predictions into four categories:

        • True Positives (TP) – Correctly predicted fraud cases.

        • True Negatives (TN) – Correctly predicted genuine jobs.

        • False Positives (FP) – Genuine jobs incorrectly labeled as fraud.

        • False Negatives (FN) – Fraud cases the model failed to detect
        
• This helps us understand model performance beyond accuracy and ensures it catches fraud with minimal false alarms.

![image](https://github.com/user-attachments/assets/eb6b775d-a080-4fea-837d-a84d0cc9db62)

Now that our model is trained and fine-tuned, we use it to predict the fraud probability of each job listing in the test dataset.

1. Use the Trained Model to Predict Fraud Probabilities:

• model.predict_proba(X_test_combined)[:, 1]
  → Returns the probability of class 1 (fraud) for each job listing in the test set.

2. Store Probabilities:

• Save the output as a new column:
    test_df['fraud_probability']

3. Convert Probabilities to Binary Predictions:

   • Apply the previously selected best_threshold:

        • If probability ≥ threshold → fraud_predicted = 1 (high likelihood of fraud)

        • If probability < threshold → fraud_predicted = 0 (likely genuine)

  • Save this as:
   test_df['fraud_predicted']

4. Result:

  • A clean column with 0s and 1s representing model predictions for fraud.

 📊 Visualizing Model Confidence
5. Histogram of Fraud Probabilities:

      • Plot all values in test_df['fraud_probability']

      • Peaks near 0 → Model is confident many jobs are real.

      • Peaks near 1 → Model has high confidence in fraud predictions.

6. KDE Line (Smooth Curve):

     • Added to the histogram to better visualize the shape of the probability distribution.

Visualizing Fraud vs Real Predictions

 7. Pie Chart of Predicted Labels:

     • Use values from test_df['fraud_predicted'] to count real vs fake predictions.

     • Provides a clear view of the percentage of job listings flagged as fake.

• Helps decision-makers or investigators understand how widespread fraud is in the test dataset according to the model.


![image](https://github.com/user-attachments/assets/2be9ef58-bd85-43cb-9967-d0ec6acd303e)


✅ Why SHAP?  

• SHAP (SHapley Additive exPlanations) is a method from game theory used to explain individual predictions made by machine learning models. It assigns each feature a contribution score (positive or negative) toward the final prediction.

1. Dense Conversion for SHAP Compatibility
   
  • SHAP requires input data in dense format.
  → We convert a subset of the training data (e.g., X_res[:100]) using .toarray().

2. SHAP Explainer Setup
   
 • We create a SHAP explainer using the trained XGBoost model and the dense data.
 → This explainer understands the model logic and computes contribution scores.

3. SHAP Values Computation
SHAP values tell how much each feature pushes a prediction toward class 1 (fraud) or class 0 (non-fraud).

4. Feature Names
   
 • We combine:

         • Top 5000 TF-IDF features from job descriptions and titles

         • Custom-engineered features like:

         • desc_len

         • word_count

         • num_digits_in_title

         • has_profile

         • suspicious_terms

         • free_email

5. Summary Plot Visualization

We generate a SHAP summary plot which shows:

 • Most influential features

 • Direction (does a feature increase or decrease fraud risk?)

 • Distribution and strength of these features across samp

 
 ![image](https://github.com/user-attachments/assets/855b0c7e-2df2-4199-85df-6ec4b8735bfa)

High-Risk Job Listings Alert

• While your model predicts all fraud probabilities, not every “fraudulent” prediction has the same confidence. A job predicted at 0.51 might just barely cross the threshold — but one predicted at 0.95 is likely very suspiciou

1. Why This Matters
   
Jobs with very high fraud probability (e.g., 0.95) are more likely to be genuinely fraudulent and may require:

   • Human verification or moderation

   • Platform filtering/blocking

   • Alerts to internal teams

 3. Implementation Overview
    
• Thresholding
A cut-off of 0.80 is used to flag high-confidence frauds.
(This value can be adjusted based on acceptable risk tolerance.)

• Filtering
From the test dataset, all listings with fraud_probability ≥ 0.8 are filtered into a separate DataFrame.

• Formatted Alert
For each high-risk listing, an alert message is generated with:

   • Title

   • Location

   • Fraud Probability (rounded to 2 decimal places)
   





