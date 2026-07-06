from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
import numpy as np
import csv
import os
import subprocess
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH      = r'E:\FAKE REVIEW DECTECTOR\Model\model.pkl'
VECTORIZER_PATH = r'E:\FAKE REVIEW DECTECTOR\Model\vectorizer.pkl'
SCALER_PATH     = r'E:\FAKE REVIEW DECTECTOR\Model\scaler.pkl'
FEEDBACK_PATH   = r'E:\FAKE REVIEW DECTECTOR\data\feedback.csv'
RETRAIN_SCRIPT  = r'E:\FAKE REVIEW DECTECTOR\Model\retrain.py'

RETRAIN_THRESHOLD = 50   # auto-retrain after this many feedback entries

# ── Load model (reloadable) ──────────────────────────────────────
def load_model():
    global model, vectorizer, scaler
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    scaler     = joblib.load(SCALER_PATH)

load_model()

stop_words = set(stopwords.words('english'))

# ── Helpers ──────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

pos_words = {'great','love','excellent','perfect','amazing','best','wonderful',
             'fantastic','outstanding','superb','happy','satisfied','good'}
neg_words = {'bad','worst','terrible','horrible','awful','waste','poor',
             'disappointed','useless','broken','fake','scam','return'}

def extract_extra_features(text):
    words = text.split()
    features = [
        text.count('!'),
        text.count('?'),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        len(words),
        np.mean([len(w) for w in words]) if words else 0,
        (len(words) - len(set(words))) / max(len(words), 1),
        len(text),
        text.count('...'),
        sum(1 for w in text.lower().split() if w in pos_words),
        sum(1 for w in text.lower().split() if w in neg_words)
    ]
    return features


def count_feedback_rows():
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
        return sum(1 for line in f) - 1   # subtract header

def run_retrain():
    """Run retrain.py in background and reload model when done."""
    try:
        subprocess.run(['python', RETRAIN_SCRIPT], check=True)
        load_model()
        print("✅ Model retrained and reloaded successfully!")
    except Exception as e:
        print(f"⚠️ Retrain failed: {e}")

# ── Schemas ──────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    review: str

class FeedbackRequest(BaseModel):
    review: str
    correct_label: str   # "Real" or "Fake"

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Fake Review Detector API is running! 🚀"}

@app.post("/predict")
def predict(request: ReviewRequest):
    clean = preprocess(request.review)
    X_tfidf = vectorizer.transform([clean])

    extra = extract_extra_features(request.review)
    extra_scaled = scaler.transform([extra])
    X_extra = csr_matrix(extra_scaled)

    X = hstack([X_tfidf, X_extra])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0]
    label = "Fake" if prediction == 1 else "Real"
    score = round(float(max(confidence)) * 100, 2)
    return {
        "review": request.review,
        "result": label,
        "confidence": f"{score}%"
    }

@app.post("/feedback")
def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Save user feedback. Auto-retrain when threshold is reached."""
    label_map = {"Real": 0, "Fake": 1}
    if request.correct_label not in label_map:
        return {"status": "error", "message": "correct_label must be 'Real' or 'Fake'"}

    # Ensure feedback file + header exist
    file_exists = os.path.exists(FEEDBACK_PATH)
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    with open(FEEDBACK_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['text', 'label'])
        writer.writerow([request.review, label_map[request.correct_label]])

    row_count = count_feedback_rows()
    should_retrain = (row_count > 0) and (row_count % RETRAIN_THRESHOLD == 0)

    if should_retrain:
        background_tasks.add_task(run_retrain)
        return {
            "status": "saved",
            "message": f"Thank you! {row_count} feedbacks collected. Retraining model in background... 🔄"
        }

    return {
        "status": "saved",
        "message": f"Thank you! Feedback saved. ({row_count}/{RETRAIN_THRESHOLD} for next retrain)"
    }

@app.get("/feedback/count")
def feedback_count():
    count = count_feedback_rows()
    return {"count": count, "threshold": RETRAIN_THRESHOLD, "next_retrain_in": RETRAIN_THRESHOLD - (count % RETRAIN_THRESHOLD)}