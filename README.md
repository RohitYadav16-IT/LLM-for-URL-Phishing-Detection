# 🛡️ Phishing URL Detection

A machine learning-based system to detect phishing URLs with explainability and LLM integration.

## 🔍 Features

- Dataset: 2.2M+ URLs labeled as phishing (0) or legitimate (1)
- ML Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Feature Extraction: URL-based heuristics (length, symbols, domains, etc.)
- LLM Support: Mixtral-8x7B for human-like risk explanations
- Explainability: SHAP for feature importance
- UI: Built with Streamlit

## 🚀 How to Run

- 1.) Extract files from original dataset.zip
- 2.) Now run dataset_combining.py to generate the combined_url_dataset.csv which will be used for model.py
- 3.) Next run the model.py for feature extraction , model evaluation and generating the best_phishing_model.pkl , scaler.pkl and feature_columns.pkl
- 4.) Run the app.py using the command :- streamlit run streamlit_app/app.py
