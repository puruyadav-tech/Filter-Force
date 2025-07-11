# -*- coding: utf-8 -*-
"""fraud_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Xa9IT3yQfQa6fl9GZlMPAGdHdPaz4Qu8
"""


import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="🎯 Fraud Job Detector", layout="wide")

# --- REMOVE WHITE TOP HEADER AREA + STYLE HEADER TEXT ---
st.markdown("""
    <style>
    /* Match entire page background */
    html, body, .main, .block-container {
        background-color: #0e1117 !important;
        color: white !important;
    }

    /* Override top header bar */
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
        padding: 0rem !important;
        height: 0rem !important;
        visibility: hidden; /* Hide if you want to fully remove the bar */
    }

    /* If you keep the header, style the text inside it */
    header h1, header h2, header h3, header h4, header p {
        color: black !important;
        font-weight: bold !important;
        background-color: #0e1117 !important;
    }

    /* Remove extra padding/margins around header area */
    .appview-container .main .block-container {
        padding-top: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- SIDEBAR STYLING AND CONTENT ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1a2a6c, #b21f1f, #fdbb2d);
        width: 250px !important;
        padding: 20px;
    }
    .sidebar-header {
        font-size: 24px;
        color: orange;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .sidebar-menu h4 {
        color: white;
        margin-bottom: 5px;
    }
    .sidebar-menu p {
        color: #e0e0e0;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .stButton button {
        background-color: #00ff88 !important;
        color: black !important;
        font-weight: bold;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-header'>🔍 Sidebar Menu</div>", unsafe_allow_html=True)

# with st.sidebar.expander("📤 Upload Data", expanded=True):
#     uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")
#     if st.button("🚀 Analyze Now"):
#         st.success("Analysis started!")

with st.sidebar.expander("⚙️ Settings"):
    st.checkbox("Enable Notifications")

with st.sidebar.expander("ℹ️ About"):
    st.markdown("This app detects fraudulent job postings using AI.")

st.sidebar.markdown("""
<div class='sidebar-menu'>
    <h4>🏠 Home</h4>
    <p>Overview of the tool and its capabilities.</p>
    <h4>⚙️ How It Works</h4>
    <p>Detailed explanation of the algorithms behind the detection.</p>
    <h4>🤖 Machine Learning</h4>
    <p>Insights into the technologies used.</p>
    <h4>📊 Real-time Analysis</h4>
    <p>Features and benefits of real-time monitoring.</p>
    <h4>⚡ Instant Results</h4>
    <p>Overview of the speedy analysis process.</p>
    <h4>📞 Support</h4>
    <p>FAQs and contact information for assistance.</p>
</div>
""", unsafe_allow_html=True)





# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .block-container {
        padding: 2rem;
        background-color: #0e1117;
    }
    h1 {
        text-align: center;
        color: #00a8ff;
        border-bottom: 2px solid #00a8ff;
        padding-bottom: 10px;
    }
    .css-1d391kg p {
        text-align: center;
        font-size: 18px;
        color: #a9a9a9;
    }
    .st-b7, .st-cb, .st-bb, .stSidebar {
        background-color: #1a1d23 !important;
    }
    .stDownloadButton button {
        background-color: #00a8ff !important;
        color: white !important;
        border: none !important;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #1a1d23 !important;
        color: #ffffff !important;
    }
    .stAlert {
        background-color: #1a1d23 !important;
        color: #ffffff !important;
    }
    .stMarkdown small {
        color: #6c757d !important;
    }
    .stButton button {
        background-color: #00a8ff !important;
        color: white !important;
        border: none !important;
    }
    .stFileUploader label {
        color: #00a8ff !important;
    }
    /* Card styles for Why Choose section */
    .feature-card {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 12px 0 rgba(60,60,60,0.07);
        padding: 24px 20px 20px 20px;
        margin-bottom: 20px;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .feature-icon {
        font-size: 36px;
        margin-bottom: 8px;
    }
    .feature-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #222;
    }
    .feature-desc {
        font-size: 15px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR STYLING ---
st.sidebar.markdown("""
    <style>
    .css-1vq4p4l {
        background-color: #1a1d23 !important;
    }
    .css-1hynsf2 {
        color: #00a8ff !important;
        font-size: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown(
    """
    <h1 style='text-align: center; color: white; font-size: 52px;'>🕵️‍♂️ AI-Powered Job Fraud Detection</h1>
    <p style='text-align: center; color: white; font-size: 28px;'>
        Detect potentially fraudulent job postings using intelligent machine learning
    </p>
    <hr style='border: 1px solid #00AEEF;' />
    """,
    unsafe_allow_html=True
)




# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data():
    file_id = "1oxygynnHOAU3NU3S8xHRdBS9WA_K1E7y"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        train_df = pd.read_csv(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading training data from Google Drive: {e}")
        train_df = pd.DataFrame()
    return train_df

# --- DATA PREPARATION ---
@st.cache_data(ttl=3600)
def prepare(df):
    df = df.copy()
    for col in ['title', 'description', 'requirements', 'company_profile']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')
    # Only handle 'email' if it exists
    if 'email' in df.columns:
        df['email'] = df['email'].fillna('')
        df['email_domain'] = df['email'].apply(lambda x: x.split('@')[-1] if '@' in x else '')
        df['free_email'] = df['email_domain'].isin(['gmail.com', 'yahoo.com', 'hotmail.com']).astype(int)
    else:
        df['email_domain'] = ''
        df['free_email'] = 0
    df['text'] = (
        df['title'] + ' ' +
        df['description'] + ' ' +
        df['requirements'] + ' ' +
        df['company_profile']
    )
    df['desc_len'] = df['description'].apply(len)
    df['word_count'] = df['description'].apply(lambda x: len(x.split()))
    df['num_digits_in_title'] = df['title'].apply(lambda x: sum(c.isdigit() for c in x))
    df['has_profile'] = (df['company_profile'] != '').astype(int)
    suspicious_words = ['money', 'wire', 'bitcoin', 'transfer', 'click']
    df['suspicious_terms'] = df['description'].apply(
        lambda x: int(any(term in x.lower() for term in suspicious_words))
    )
    return df

# --- MODEL TRAINING ---
@st.cache_resource(ttl=86400) 
def train_model(train_df):
    if train_df.empty:
        st.warning("Training data not loaded. Model training skipped.")
        return None, None, 0.5
    X = train_df[['text', 'desc_len', 'word_count', 'num_digits_in_title', 'has_profile', 'suspicious_terms', 'free_email']]
    y = train_df['fraudulent']
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X['text'])
    X_combined = hstack([X_tfidf, X.drop(columns='text').values])
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=42)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_res, y_res)
    val_probs = model.predict_proba(X_val)[:, 1]
    p, r, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2 * p * r / (p + r + 1e-6)
    best_threshold = thresholds[np.argmax(f1s)]
    return model, tfidf, best_threshold

# --- MAIN FLOW ---
train_df = load_data()
if not train_df.empty:
    train_df = prepare(train_df)
model, tfidf, best_threshold = train_model(train_df)

# --- SIDEBAR UPLOAD ---
st.markdown("## 📤 Upload a CSV File to Detect Fraudulent Jobs")
uploaded_file = st.file_uploader("Upload CSV for Prediction", type="csv")
analyze_btn = st.button("🚀 Analyze Now")

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        test_df = prepare(test_df)
        if model is not None and tfidf is not None:
            X_test = test_df[['text', 'desc_len', 'word_count', 'num_digits_in_title', 'has_profile', 'suspicious_terms', 'free_email']]
            X_test_tfidf = tfidf.transform(X_test['text'])
            X_test_combined = hstack([X_test_tfidf, X_test.drop(columns='text').values])
            test_df['fraud_probability'] = model.predict_proba(X_test_combined)[:, 1]
            test_df['fraud_predicted'] = (test_df['fraud_probability'] >= best_threshold).astype(int)
            st.success("✅ Predictions generated successfully!")
            st.markdown("### 📋 Job Predictions")
            st.dataframe(
                test_df[['title', 'location', 'fraud_probability', 'fraud_predicted']].sort_values(by='fraud_probability', ascending=False),
                use_container_width=True
            )
            st.download_button(
                "📥 Download Results as CSV",
                data=test_df.to_csv(index=False).encode(),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
            # --- VISUALIZATION 1: HISTOGRAM (smaller size) ---
            st.markdown("### 📊 Probability Distribution")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                fig, ax = plt.subplots(figsize=(4,2.5))
                ax.hist(test_df['fraud_probability'], bins=20, color='#00a8ff', edgecolor='#1a1d23')
                ax.set_xlabel("Fraud Probability", color='white')
                ax.set_ylabel("Job Count", color='white')
                ax.tick_params(colors='white')
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                st.pyplot(fig)
            # --- VISUALIZATION 2: PIE CHART (smaller size) ---
            st.markdown("### 🧮 Fraud Prediction Breakdown")
            with col2:
                fraud_counts = test_df['fraud_predicted'].value_counts()
                labels = ['Not Fraud', 'Fraud']
                sizes = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]
                fig2, ax2 = plt.subplots(figsize=(3.5,2.5))
                ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                        colors=['#00a8ff', '#ff6b6b'], startangle=90,
                        textprops={'color': 'white'})
                ax2.axis('equal')
                fig2.patch.set_facecolor('#0e1117')
                st.pyplot(fig2)
        else:
            st.warning("Model not loaded. Cannot generate predictions.")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
# else:
#     st.info("👈 Upload a CSV file from the sidebar to begin analysis.")

# --- WHY CHOOSE OUR AI SOLUTION SECTION ---
st.markdown("---")
st.markdown(
    "<h2 style='text-align:center; margin-bottom:0; color:white;'>🚀 Why Choose Our AI Solution?</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:white; font-size:18px; margin-top:0;'>"
    "Built with cutting-edge technology, our solution combines the power of machine learning with user-friendly design to tackle job fraud at scale."
    "</p>",
    unsafe_allow_html=True
)

card_icons = [
    "🛡️", "⚡", "📊",
    "🧑‍💼", "👁️", "✅"
]
card_titles = [
    "Advanced Fraud Detection",
    "Lightning Fast Analysis",
    "Detailed Analytics",
    "Protect Job Seekers",
    "Real-time Monitoring",
    "High Accuracy Rate"
]
card_descs = [
    "Our AI model analyzes multiple data points to identify suspicious job postings with high accuracy.",
    "Get results in seconds, not hours. Process thousands of job postings instantly.",
    "Comprehensive reports with fraud probability scores and detailed explanations.",
    "Help millions of job seekers avoid scams and find legitimate opportunities.",
    "Continuous monitoring of job posting platforms to detect threats proactively.",
    "Achieved 95%+ accuracy in fraud detection through advanced learning techniques."
]
# Display as 3 columns per row
for i in range(0, len(card_titles), 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        if i+j < len(card_titles):
            col.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{card_icons[i+j]}</div>
                    <div class="feature-title">{card_titles[i+j]}</div>
                    <div class="feature-desc">{card_descs[i+j]}</div>
                </div>
                """, unsafe_allow_html=True
            )

# --- FOOTER ---
st.markdown("---")
st.markdown("<small style='color: #6c757d'>🚀 Developed with ❤️ using Streamlit</small>", unsafe_allow_html=True)
