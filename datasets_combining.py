import pandas as pd

# === Load Data Files ===
url_dataset = pd.read_csv('original dataset/URL dataset.csv')
new_data = pd.read_csv('original dataset/new_data_urls.csv')
top_1m = pd.read_csv('original dataset/top-1m.csv', header=None, names=['rank', 'domain'])
phishing_df = pd.read_csv('original dataset/dataset_phishing.csv')

# === Process URL dataset.csv ===
url_dataset['status'] = url_dataset['type'].map({'phishing': 0, 'legitimate': 1})
url_dataset['url'] = url_dataset['url'].astype(str)
url_dataset_clean = url_dataset[['url', 'status']]

# === Process new_data_urls.csv ===
new_data['url'] = new_data['url'].astype(str)
new_data['url'] = new_data['url'].apply(
    lambda x: x if x.startswith(('http://', 'https://')) else 'http://' + x
)
new_data_clean = new_data[['url', 'status']]

# === Process top-1m.csv ===
top_1m['url'] = 'http://' + top_1m['domain'].astype(str)
top_1m['status'] = 1  # Legitimate
top_1m_clean = top_1m[['url', 'status']]

# === Process dataset_phishing.csv ===
phishing_df['url'] = phishing_df['url'].astype(str)
phishing_df['status'] = phishing_df['status'].map({'phishing': 0, 'legitimate': 1})
phishing_clean = phishing_df[['url', 'status']]

# === Combine all datasets ===
combined_df = pd.concat(
    [url_dataset_clean, new_data_clean, top_1m_clean, phishing_clean],
    ignore_index=True
)

# === Shuffle the combined dataset ===
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Save to CSV ===
combined_df.to_csv('combined_url_dataset.csv', index=False)

# === Print Summary ===
print("✅ Combined Dataset Saved as 'combined_url_dataset.csv'")
print(f"Total Rows: {len(combined_df)}")
print("Class Balance (%):")
print(combined_df['status'].value_counts(normalize=True) * 100)
