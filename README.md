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

## Screenshots
<img width="1917" height="968" alt="Screenshot 2025-05-14 095558" src="https://github.com/user-attachments/assets/89a8cb57-d3ab-4d44-89df-406821c6e16e" />

<img width="1910" height="857" alt="Screenshot 2025-05-14 095709" src="https://github.com/user-attachments/assets/29f47e12-84f8-4f49-ac81-688434fd1229" />

<img width="1918" height="968" alt="Screenshot 2025-05-14 095802" src="https://github.com/user-attachments/assets/25946ffe-451c-4b04-8ae4-6d84101f2247" />

<img width="1918" height="962" alt="Screenshot 2025-05-14 095947" src="https://github.com/user-attachments/assets/5aa22049-191b-4793-88af-16305309484b" />

