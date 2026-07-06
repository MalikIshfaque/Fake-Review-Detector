"""
============================================================
FAKE REVIEW DETECTOR - v2 Training Script
============================================================
Datasets Used:
  1. theArijitDas/Fake-Reviews-Dataset  (40,000+ Amazon reviews, real vs AI-fake)
  2. debojit01/fake-review-dataset       (Home & Kitchen Amazon reviews)
  3. Existing local dataset              (Hotel reviews)

Model: Logistic Regression with TF-IDF + hand-crafted features
============================================================
"""

import pandas as pd
import numpy as np
import re
import csv
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("=" * 60)
print("  FAKE REVIEW DETECTOR - v2 Training")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load datasets
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading datasets...")

frames = []

# ── 1a. Existing local dataset (hotel reviews) ──
try:
    df_local = pd.read_csv(
        r'E:\FAKE REVIEW DECTECTOR\data\reviews.txt',
        sep='\t',
        engine='python',
        on_bad_lines='skip',
        quoting=csv.QUOTE_NONE
    )
    df_local = df_local.rename(columns={'REVIEW_TEXT': 'text', 'LABEL': 'label'})
    df_local['label'] = df_local['label'].map({'__label1__': 1, '__label2__': 0})
    df_local = df_local.dropna(subset=['text', 'label'])
    df_local['label'] = df_local['label'].astype(int)
    df_local = df_local[['text', 'label']]
    print(f"  Local dataset: {len(df_local)} rows")
    frames.append(df_local)
except Exception as e:
    print(f"  [WARNING] Local dataset error: {e}")

# ── 1b. Hugging Face datasets ──
try:
    from datasets import load_dataset

    # Dataset 1: theArijitDas (40k Amazon reviews)
    print("  Downloading theArijitDas/Fake-Reviews-Dataset from HuggingFace...")
    ds1 = load_dataset("theArijitDas/Fake-Reviews-Dataset", split="train")
    df1 = ds1.to_pandas()
    print(f"  Columns: {df1.columns.tolist()}")

    # Find text column
    text_col = next((c for c in df1.columns if 'text' in c.lower() or 'review' in c.lower()), df1.columns[0])
    label_col = next((c for c in df1.columns if 'label' in c.lower() or 'fake' in c.lower() or 'class' in c.lower()), None)

    df1 = df1.rename(columns={text_col: 'text'})
    if label_col and label_col != 'text':
        df1 = df1.rename(columns={label_col: 'label'})
        if df1['label'].dtype == object:
            df1['label'] = df1['label'].map({'OR': 0, 'CG': 1, 'Real': 0, 'Fake': 1, 'real': 0, 'fake': 1, '0': 0, '1': 1})
        df1['label'] = pd.to_numeric(df1['label'], errors='coerce')
        df1 = df1[['text', 'label']].dropna()
        df1['label'] = df1['label'].astype(int)
        print(f"  HuggingFace Dataset 1: {len(df1)} rows | Labels: {df1['label'].value_counts().to_dict()}")
        frames.append(df1)

except Exception as e:
    print(f"  [WARNING] HuggingFace dataset 1 error: {e}")
    print("  Install with: pip install datasets")

try:
    from datasets import load_dataset

    # Dataset 2: debojit01 (Home & Kitchen Amazon reviews)
    print("  Downloading debojit01/fake-review-dataset from HuggingFace...")
    ds2 = load_dataset("debojit01/fake-review-dataset", split="train")
    df2 = ds2.to_pandas()
    print(f"  Columns: {df2.columns.tolist()}")

    text_col = next((c for c in df2.columns if 'text' in c.lower() or 'review' in c.lower()), df2.columns[0])
    label_col = next((c for c in df2.columns if 'label' in c.lower() or 'fake' in c.lower()), None)

    if label_col:
        df2 = df2.rename(columns={text_col: 'text', label_col: 'label'})
        if df2['label'].dtype == object:
            df2['label'] = df2['label'].map({'OR': 0, 'CG': 1, 'Real': 0, 'Fake': 1, 'real': 0, 'fake': 1, '0': 0, '1': 1})
        df2['label'] = pd.to_numeric(df2['label'], errors='coerce')
        df2 = df2[['text', 'label']].dropna()
        df2['label'] = df2['label'].astype(int)
        print(f"  HuggingFace Dataset 2: {len(df2)} rows | Labels: {df2['label'].value_counts().to_dict()}")
        frames.append(df2)

except Exception as e:
    print(f"  [WARNING] HuggingFace dataset 2 error: {e}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Combine all datasets
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Combining datasets...")

if not frames:
    raise RuntimeError("No datasets loaded! Check errors above.")

df = pd.concat(frames, ignore_index=True)
print(f"  Total combined rows: {len(df)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Clean & Generalize
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Cleaning dataset...")

df['text'] = df['text'].astype(str).str.strip()
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
df = df[df['text'].str.len() > 5]
df = df[df['text'].apply(lambda x: len(x.split())) >= 3]

before = len(df)
df = df.drop_duplicates(subset=['text'])
print(f"  Duplicates removed: {before - len(df)}")

df = df[df['label'].isin([0, 1])]
print(f"  Final clean size: {len(df)}")
print(f"\n  Class balance:")
print(df['label'].value_counts())

# ─────────────────────────────────────────────────────────────
# STEP 4: Preprocess text
# ─────────────────────────────────────────────────────────────
print("\n[STEP 4] Preprocessing text...")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)
df = df[df['clean_text'].str.len() > 0]
print(f"  After preprocessing: {len(df)} rows")

# ─────────────────────────────────────────────────────────────
# STEP 5: Extract hand-crafted features (10 features)
# ─────────────────────────────────────────────────────────────
pos_words = {'great','love','excellent','perfect','amazing','best','wonderful',
             'fantastic','outstanding','superb','happy','satisfied','good'}
neg_words = {'bad','worst','terrible','horrible','awful','waste','poor',
             'disappointed','useless','broken','fake','scam','return'}

def extract_features(texts):
    feats = []
    for text in texts:
        words = text.split()
        feats.append([
            text.count('!'),
            text.count('?'),
            sum(1 for c in text if c.isupper()) / max(len(text), 1),
            len(words),
            np.mean([len(w) for w in words]) if words else 0,
            (len(words) - len(set(words))) / max(len(words), 1),
            len(text),
            text.count('...'),
            sum(1 for w in text.lower().split() if w in pos_words),
            sum(1 for w in text.lower().split() if w in neg_words),
        ])
    return np.array(feats)

# ─────────────────────────────────────────────────────────────
# STEP 6: TF-IDF Vectorization
# ─────────────────────────────────────────────────────────────
print("\n[STEP 6] Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    analyzer='word'
)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
print(f"  TF-IDF shape: {X_tfidf.shape}")

extra_features = extract_features(df['text'].tolist())
scaler = StandardScaler()
extra_scaled = scaler.fit_transform(extra_features)
X_extra = csr_matrix(extra_scaled)

X = hstack([X_tfidf, X_extra])
y = df['label']

# ─────────────────────────────────────────────────────────────
# STEP 7: Train/Test Split
# ─────────────────────────────────────────────────────────────
print("\n[STEP 7] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────
# STEP 8: Train Model
# ─────────────────────────────────────────────────────────────
print("\n[STEP 8] Training model...")

model = LogisticRegression(
    C=1.0,
    max_iter=5000,
    class_weight='balanced',
    solver='lbfgs',
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  Model trained!")

# ─────────────────────────────────────────────────────────────
# STEP 9: Evaluate
# ─────────────────────────────────────────────────────────────
print("\n[STEP 9] Evaluating...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n  Accuracy: {acc:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
print("  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sanity check
print("\n  Sanity Check:")
test_reviews = [
    "I really liked this product, works great and fast delivery.",
    "Amazing product! Best purchase ever! Buy now! 5 stars!!!",
    "The quality is decent but packaging was slightly damaged.",
    "Totally fake product, waste of money, do not buy!!!",
    "Good product, does what it says, would recommend.",
]
for review in test_reviews:
    clean = preprocess(review)
    xt = vectorizer.transform([clean])
    ef = scaler.transform(extract_features([review]))
    xc = hstack([xt, csr_matrix(ef)])
    pred = model.predict(xc)[0]
    prob = model.predict_proba(xc)[0]
    label = "FAKE" if pred == 1 else "REAL"
    confidence = round(float(max(prob)) * 100, 1)
    print(f"  [{label} {confidence}%] \"{review[:65]}\"")

# ─────────────────────────────────────────────────────────────
# STEP 10: Save
# ─────────────────────────────────────────────────────────────
print("\n[STEP 10] Saving model...")
joblib.dump(model,      r'E:\FAKE REVIEW DECTECTOR\Model\model.pkl')
joblib.dump(vectorizer, r'E:\FAKE REVIEW DECTECTOR\Model\vectorizer.pkl')
joblib.dump(scaler,     r'E:\FAKE REVIEW DECTECTOR\Model\scaler.pkl')
print("  model.pkl saved")
print("  vectorizer.pkl saved")
print("  scaler.pkl saved")
print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
