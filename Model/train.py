import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
import csv

df = pd.read_csv(
    r'E:\FAKE REVIEW DECTECTOR\data\reviews.txt',
    sep='\t',
    engine='python',
    on_bad_lines='skip',
    quoting=csv.QUOTE_NONE
)

print("Columns found:", df.columns.tolist())
print("Unique LABEL values:", df['LABEL'].unique())
print(df.head())

df = df.rename(columns={'REVIEW_TEXT': 'text', 'LABEL': 'label'})

# __label1__ = Deceptive (FAKE), __label2__ = Truthful (REAL)
print("Unique labels before mapping:", df['label'].unique())
df['label'] = df['label'].map({'__label1__': 1, '__label2__': 0})
print("Null labels after mapping:", df['label'].isnull().sum())
df = df.dropna(subset=['label'])  # Remove any unmapped labels
df['label'] = df['label'].astype(int)

print(f"Raw rows: {len(df)}")

# ─────────────────────────────────────────────
# STEP 2: Clean Dataset (NEW)
# ─────────────────────────────────────────────
# Drop missing values
df = df.dropna(subset=['text', 'label'])
print(f"After dropping missing values: {len(df)}")

# Normalize whitespace (so "Great!!" and "Great!!  " aren't treated differently)
df['text'] = df['text'].astype(str).str.strip()
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

# Drop empty strings after stripping
df = df[df['text'].str.len() > 0]

# Drop exact duplicate reviews (this is the big one — prevents train/test leakage)
before = len(df)
df = df.drop_duplicates(subset=['text'])
after = len(df)
print(f"Duplicates removed: {before - after}")
print(f"Final dataset size: {after}")

# Check class balance
print(f"\nClass balance:")
print(df['label'].value_counts())
print(f"Fake %: {(df['label'].sum() / len(df)) * 100:.2f}%")
print(f"Real %: {((df['label']==0).sum() / len(df)) * 100:.2f}%")

# ─────────────────────────────────────────────
# STEP 3: Extra Features
# ─────────────────────────────────────────────
def extract_extra_features(df):
    features = pd.DataFrame()
    features['exclamation'] = df['text'].str.count('!')
    features['caps_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    features['word_count'] = df['text'].apply(lambda x: len(x.split()))
    features['avg_word_len'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    features['repeated_words'] = df['text'].apply(
        lambda x: len(x.split()) - len(set(x.split()))
    )
    return features.values

# ─────────────────────────────────────────────
# STEP 4: Preprocessing
# ─────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)

# Drop rows that became empty after preprocessing (e.g. text was only punctuation/numbers)
df = df[df['clean_text'].str.len() > 0]
print(f"\nAfter removing empty-after-preprocessing rows: {len(df)}")
print("Preprocessing done!")
print("\nSample cleaned texts (sanity check):")
print(df[['label', 'clean_text']].sample(3, random_state=1))

# ─────────────────────────────────────────────
# STEP 5: TF-IDF + Extra Features Combine
# ─────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=1,
    analyzer='word'
)
# CRITICAL FIX: Train vectorizer on SAME text that backend will send (clean_text)
X_tfidf = vectorizer.fit_transform(df['clean_text'])

extra_features = extract_extra_features(df)
scaler = StandardScaler()
extra_features_scaled = scaler.fit_transform(extra_features)
X_extra = csr_matrix(extra_features_scaled)

X = hstack([X_tfidf, X_extra])
y = df['label']

# ─────────────────────────────────────────────
# STEP 6: Train Test Split (stratified so both sets keep the same class ratio)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# STEP 7: Train Model
# ─────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', solver='lbfgs')
model.fit(X_train, y_train)
print("\nModel trained!")

# Sanity check - print some predictions on real-looking reviews
sample_real = ["I really liked this product it works great",
               "Good quality and fast shipping"]
for s in sample_real:
    from nltk.corpus import stopwords as sw
    stop = set(sw.words('english'))
    c = ' '.join([w for w in re.sub(r'[^a-z\s]','',s.lower()).split() if w not in stop])
    xt = vectorizer.transform([c])
    ef = scaler.transform([[s.count('!'), sum(1 for ch in s if ch.isupper())/max(len(s),1),
                            len(s.split()), np.mean([len(w) for w in s.split()]),
                            len(s.split())-len(set(s.split()))]])
    xc = hstack([xt, csr_matrix(ef)])
    pred = model.predict(xc)[0]
    print(f"  '{s}' → {'FAKE' if pred==1 else 'REAL'}")

# ─────────────────────────────────────────────
# STEP 8: Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# ─────────────────────────────────────────────
# STEP 9: Save
# ─────────────────────────────────────────────
joblib.dump(model, r'E:\FAKE REVIEW DECTECTOR\Model\model.pkl')
joblib.dump(vectorizer, r'E:\FAKE REVIEW DECTECTOR\Model\vectorizer.pkl')
joblib.dump(scaler, r'E:\FAKE REVIEW DECTECTOR\Model\scaler.pkl')
print("Model saved!")