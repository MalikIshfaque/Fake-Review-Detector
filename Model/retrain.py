"""
retrain.py — Lightweight retrain using original data + feedback.csv
Runs automatically when enough feedback is collected.
"""
import pandas as pd
import numpy as np
import re
import csv
import nltk
import joblib
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("=" * 50)
print("  AUTO-RETRAIN TRIGGERED")
print("=" * 50)

frames = []

# ── Load original data ────────────────────────────────────────────
try:
    df_orig = pd.read_csv(
        r'E:\FAKE REVIEW DECTECTOR\data\reviews.txt',
        sep='\t', engine='python',
        on_bad_lines='skip', quoting=csv.QUOTE_NONE
    )
    df_orig = df_orig.rename(columns={'REVIEW_TEXT': 'text', 'LABEL': 'label'})
    df_orig['label'] = df_orig['label'].map({'__label1__': 1, '__label2__': 0})
    df_orig = df_orig.dropna(subset=['text', 'label'])
    df_orig['label'] = df_orig['label'].astype(int)
    df_orig = df_orig[['text', 'label']]
    frames.append(df_orig)
    print(f"  Original data: {len(df_orig)} rows")
except Exception as e:
    print(f"  [WARNING] Original data error: {e}")

# ── Load feedback data ────────────────────────────────────────────
FEEDBACK_PATH = r'E:\FAKE REVIEW DECTECTOR\data\feedback.csv'
if os.path.exists(FEEDBACK_PATH):
    df_fb = pd.read_csv(FEEDBACK_PATH)
    df_fb = df_fb.dropna(subset=['text', 'label'])
    df_fb['label'] = df_fb['label'].astype(int)
    frames.append(df_fb)
    print(f"  Feedback data: {len(df_fb)} rows")
else:
    print("  No feedback data found.")

if not frames:
    raise RuntimeError("No data available for retraining!")

# ── Combine ───────────────────────────────────────────────────────
df = pd.concat(frames, ignore_index=True)
df['text'] = df['text'].astype(str).str.strip()
df = df.drop_duplicates(subset=['text'])
df = df[df['label'].isin([0, 1])]
print(f"  Total rows: {len(df)}")

# ── Preprocess ────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)
df = df[df['clean_text'].str.len() > 0]

# ── Features ──────────────────────────────────────────────────────
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

vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,3), sublinear_tf=True, min_df=1)
X_tfidf = vectorizer.fit_transform(df['clean_text'])

scaler = StandardScaler()
X_extra = csr_matrix(scaler.fit_transform(extract_features(df['text'].tolist())))

X = hstack([X_tfidf, X_extra])
y = df['label']

# ── Train ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', solver='lbfgs')
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test)) * 100
print(f"  Accuracy: {acc:.2f}%")

# ── Save ──────────────────────────────────────────────────────────
joblib.dump(model,      r'E:\FAKE REVIEW DECTECTOR\Model\model.pkl')
joblib.dump(vectorizer, r'E:\FAKE REVIEW DECTECTOR\Model\vectorizer.pkl')
joblib.dump(scaler,     r'E:\FAKE REVIEW DECTECTOR\Model\scaler.pkl')
print("  model.pkl saved")
print("  vectorizer.pkl saved")
print("  scaler.pkl saved")
print("=" * 50)
print(f"  Retrain Complete! Accuracy: {acc:.2f}%")
print("=" * 50)
