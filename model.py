import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import time

# === Load Dataset ===
df = pd.read_csv('combined_url_dataset.csv')
print(f"✅ Loaded dataset with {len(df)} URLs.")

# === Entropy Calculation ===
def calculate_entropy(string):
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    return -sum([p * np.log2(p) for p in prob]) if string else 0

# === Enhanced Safe Feature Extraction ===
def safe_extract_features(url):
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        path = parsed.path
        query = parsed.query

        subdomains = ext.subdomain.split('.') if ext.subdomain else []

        suspicious_tlds = {'xyz', 'top', 'club', 'online', 'site', 'pw', 'info', 'biz'}

        return {
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
    except Exception as e:
        print(f"[!] Skipping invalid URL: {url} | Error: {e}")
        return None

# === Extract Features ===
features_list = []
valid_indices = []

for idx, url in enumerate(df['url']):
    features = safe_extract_features(url)
    if features:
        features_list.append(features)
        valid_indices.append(idx)

features_df = pd.DataFrame(features_list)
features_df['status'] = df.loc[valid_indices, 'status'].values
print(f"✅ Valid URLs used: {len(features_df)}")

# === Reduce TLD Dimensionality ===
top_tlds = features_df['tld'].value_counts().nlargest(20).index.tolist()
features_df['tld'] = features_df['tld'].apply(lambda x: x if x in top_tlds else 'other')
features_df = pd.get_dummies(features_df, columns=['tld'], drop_first=True)
print(f"✅ Total Columns After TLD Reduction: {features_df.shape[1]}")

# === Train-Test Split ===
X = features_df.drop('status', axis=1).astype('float32')
y = features_df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define ML Models ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.5, random_state=42),
    "LightGBM": lgb.LGBMClassifier(class_weight='balanced', random_state=42),
    "CatBoost": cb.CatBoostClassifier(verbose=0, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

# === Train & Evaluate with Training Time ===
results = []

for name, model in models.items():
    print(f"\n====================")
    print(f"🔹 Training: {name}")
    print(f"====================")

    try:
        start_time = time.time()

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        end_time = time.time()
        train_time = end_time - start_time

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(classification_report(y_test, y_pred))
        print(f"✅ Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | ROC AUC: {auc:.4f} | Training Time: {train_time:.2f}s")

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': auc,
            'Training Time (s)': train_time
        })

    except Exception as e:
        print(f"[!] Error training {name}: {e}")

# === Results Summary ===
results_df = pd.DataFrame(results).sort_values(by='ROC AUC', ascending=False)
print("\n=== 📊 Model Performance Summary ===")
print(results_df)

csv_filename = 'model_evaluation_metrics.csv'
results_df.to_csv(csv_filename, index=False)
print(f"📂 Evaluation metrics saved to: {csv_filename}")

# === Plot Model Performance with Training Time ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar width and index
bar_width = 0.15
index = np.arange(len(results_df['Model']))

# Bar Chart for Metrics
ax1.bar(index, results_df['Accuracy'], bar_width, label='Accuracy', color='#1f77b4')
ax1.bar(index + bar_width, results_df['Precision'], bar_width, label='Precision', color='#ff7f0e')
ax1.bar(index + 2 * bar_width, results_df['Recall'], bar_width, label='Recall', color='#2ca02c')
ax1.bar(index + 3 * bar_width, results_df['F1 Score'], bar_width, label='F1 Score', color='#17becf')
ax1.bar(index + 4 * bar_width, results_df['ROC AUC'], bar_width, label='ROC AUC', color='#9467bd')

# Formatting
ax1.set_xlabel('Model')
ax1.set_ylabel('Score (0-1)')
ax1.set_xticks(index + 2 * bar_width)
ax1.set_xticklabels(results_df['Model'], rotation=30)
ax1.set_title("Model Evaluation Metrics with Training Time")
ax1.legend(loc='upper left')

# Secondary Y-Axis for Training Time (Line Chart)
ax2 = ax1.twinx()
ax2.plot(index + 2 * bar_width, results_df['Training Time (s)'], marker='o', linestyle='-', color='green', linewidth=2, label='Training Time (s)')
ax2.set_ylabel('Training Time (s)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('model_evaluation_graph.png')
plt.show()

print(f"📂 Graph saved to: model_evaluation_graph.png")


# === Save Best Model ===
best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best Performing Model: {best_model_name}")
best_model = models[best_model_name]

joblib.dump(best_model, 'best_phishing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print(f"\n📂 Saved: best_phishing_model.pkl")
print(f"📂 Saved: scaler.pkl")
print(f"📂 Saved: feature_columns.pkl")
