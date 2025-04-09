import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import shap
import plotly.graph_objects as go
from urllib.parse import urlparse
import tldextract
import requests

# === Hugging Face API Settings ===
HF_API_KEY = "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"
HF_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# === Streamlit Config ===
st.set_page_config(page_title="🔍 URL Phishing Detector + AI", page_icon="🔐", layout="centered")

# === Custom CSS Styling ===
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stTextInput > div > div > input { border-radius: 10px; border: 1px solid #ccc; }
    .stButton > button { border-radius: 10px; background-color: #4CAF50; color: white; }
    .stButton > button:hover { background-color: #45a049; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .small-text { font-size: 16px; line-height: 1.6; }
    .risk-box { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# === Load Model, Scaler, and Features ===
model = joblib.load('best_phishing_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
top_tlds = [col.replace('tld_', '') for col in feature_columns if col.startswith('tld_')]

# === Trusted Domains List ===
trusted_domains = ["facebook.com", "google.com", "amazon.com"]
suspicious_tlds = {'xyz', 'top', 'club', 'online', 'site', 'pw', 'info', 'biz'}

# === Utility Functions ===
def calculate_entropy(string):
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    return -sum([p * np.log2(p) for p in prob]) if string else 0

def extract_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    path = parsed.path
    query = parsed.query
    subdomains = ext.subdomain.split('.') if ext.subdomain else []

    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_slashes': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'num_subdomains': len(subdomains),
        'subdomain_length': len(ext.subdomain),
        'domain_length': len(ext.domain),
        'tld_length': len(ext.suffix),
        'path_length': len(path),
        'query_length': len(query),
        'num_parameters': query.count('='),
        'has_ip': bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', parsed.netloc)),
        'has_https_token': 'https' in parsed.netloc.lower(),
        'has_at_symbol': '@' in url,
        'has_login_keyword': bool(re.search(r'login|secure|account|update|verify', url.lower())),
        'has_port_in_url': ':' in parsed.netloc and parsed.netloc.split(':')[-1].isdigit(),
        'starts_with_http': url.lower().startswith('http'),
        'starts_with_www': parsed.netloc.lower().startswith('www'),
        'entropy': calculate_entropy(url),
        'tld': ext.suffix,
        'has_suspicious_tld': ext.suffix.lower() in suspicious_tlds,
        'is_https': parsed.scheme == 'https'
    }

    for tld in top_tlds:
        features[f'tld_{tld}'] = 1 if ext.suffix == tld else 0
    features['tld_other'] = 1 if ext.suffix not in top_tlds else 0

    features.pop('tld')  # Remove raw TLD

    for col in feature_columns:
        if col not in features:
            features[col] = 0

    return pd.DataFrame([features])[feature_columns]

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

def generate_explanation_mixtral(prompt_text):
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 150,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }

    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text'].strip()
    else:
        return f"⚠️ Hugging Face API Error: {response.status_code} - {response.text}"

# === App UI ===
st.title("🔍 URL Phishing Detector + AI Explanation")
st.markdown("Detect whether a URL is **Legitimate** or **Phishing** using machine learning and AI explanations.")

url_input = st.text_input("🔗 Enter a URL", placeholder="https://example.com/login")

if st.button("🚦 Analyze URL"):
    if url_input:
        with st.spinner("Analyzing URL and generating explanation..."):
            parsed_url = urlparse(url_input)
            domain = parsed_url.netloc.lower()

            if any(trusted in domain for trusted in trusted_domains):
                prob = 0.0
                risk_label = "✅ **Trusted Domain**"
                color = "#4CAF50"
            else:
                features_df = extract_features(url_input)
                scaled_features = scaler.transform(features_df)
                prob = model.predict_proba(scaled_features)[0][1]

                if prob >= 0.8:
                    risk_label = "🚨 **High Risk: Likely Phishing**"
                    color = "#ff4d4d"
                elif prob >= 0.5:
                    risk_label = "⚠️ **Moderate Risk: Suspicious**"
                    color = "#ffa500"
                else:
                    risk_label = "✅ **Low Risk: Likely Safe**"
                    color = "#4CAF50"

            # === Risk Level Container ===
            with st.container():
                st.markdown(f"""
                <div style="background-color:{color};padding:10px;border-radius:10px;color:white;font-weight:bold;text-align:center;" class="risk-box">
                    {risk_label}<br>Probability of phishing: {prob:.2%}
                </div>
                """, unsafe_allow_html=True)

            # === SHAP Feature Impact with Mixtral Explanation ===
            if prob > 0.5:
                shap_values = explainer.shap_values(scaled_features)
                impact_data = sorted(
                    zip(features_df.columns, features_df.iloc[0], shap_values[0]),
                    key=lambda x: abs(x[2]), reverse=True
                )

                with st.container():
                    st.markdown("### 📊 SHAP Feature Impact with AI Explanation")
                    for feature_name, feature_value, shap_value in impact_data[:10]:
                        feature_prompt = f"""
Explain how the feature **{feature_name}**, with value **{feature_value:.4f}**, and SHAP impact **{'+' if shap_value >= 0 else ''}{shap_value:.4f}**, contributes to a phishing prediction. Keep it short, 1-2 sentences.
""".strip()
                        explanation_text = generate_explanation_mixtral(feature_prompt)

                        st.markdown(f"""
                        <div class="small-text">
                        <b>Feature:</b> <code>{feature_name}</code><br>
                        <b>Value:</b> <code>{feature_value:.4f}</code><br>
                        <b>SHAP Impact:</b> <code>{'+' if shap_value >= 0 else ''}{shap_value:.4f}</code><br>
                        🧠 <b>AI Explanation:</b> {explanation_text}
                        </div>
                        <hr style="margin:4px 0;">
                        """, unsafe_allow_html=True)
                        
            # === Overall Mixtral AI Explanation ===
            if prob > 0.5:
             with st.container():
                st.markdown("### 🧠 Mixtral AI Overall Explanation")
                overall_prompt = f"""
The URL below was analyzed and predicted as **{risk_label}** with a phishing confidence of **{prob*100:.2f}%**.
URL: {url_input}

Explain why this URL received this classification. Consider its structure, presence of suspicious patterns, and any common phishing indicators. Keep the explanation accurate, concise, and user-friendly.
""".strip()
                overall_explanation = generate_explanation_mixtral(overall_prompt)
                st.markdown(f"<div class='small-text'>{overall_explanation}</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a valid URL.")
