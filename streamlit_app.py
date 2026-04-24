import streamlit as st
import joblib
import re
import json
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS)

st.set_page_config(
    page_title="Fake News Dashboard",
    layout="wide",
    page_icon="📰"
)

log_model = joblib.load("outputs/models/log_model.pkl")
vectorizer = joblib.load("outputs/models/vectorizer.pkl")

with open("outputs/results/all_metrics.json") as f:
    metrics = json.load(f)

log_acc = metrics["logistic_regression"]["accuracy"]
lstm_acc = metrics["lstm"]["accuracy"]

with open("outputs/models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

device = torch.device("cpu")

lstm_model = LSTMModel(len(vocab) + 1)
lstm_model.load_state_dict(torch.load("outputs/models/lstm_model.pth", map_location=device))
lstm_model.eval()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'reuters|21st century wire', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def to_seq(text):
    words = clean_text(text).split()
    seq = [vocab.get(w, 0) for w in words]
    seq = seq[:200] + [0] * (200 - len(seq))
    return np.array(seq)

def predict_lstm(text):
    seq = torch.tensor(to_seq(text)).unsqueeze(0)

    with torch.no_grad():
        logit = lstm_model(seq).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    return int(prob > 0.5), prob

# =========================
# LOGISTIC EXPLANATION
# =========================
def explain(text):
    vec = vectorizer.transform([clean_text(text)])
    names = vectorizer.get_feature_names_out()
    coefs = log_model.coef_[0]

    idx = vec.nonzero()[1]

    scores = [
        (names[i], coefs[i] * vec[0, i])
        for i in idx
    ]

    return sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:8]

# =========================
# HARD TEST MODE
# =========================
def hard_test_mode(text):
    hard_keywords = [
        "scientists", "report", "study", "data", "analysis",
        "official", "government", "research", "evidence",
        "according to", "published", "journal"
    ]

    return any(w in text.lower() for w in hard_keywords)

# =========================
# HEURISTIC MODEL
# =========================
def heuristic_label(text):
    fake_signals = [
        "breaking", "shocking", "unbelievable",
        "you won't believe", "secret", "miracle",
        "urgent", "exposed"
    ]

    real_signals = [
        "according to", "reported", "stated",
        "announced", "data shows", "research", "study"
    ]

    fake_score = sum(1 for w in fake_signals if w in text.lower())
    real_score = sum(1 for w in real_signals if w in text.lower())

    if fake_score == real_score:
        return 1

    return 0 if fake_score > real_score else 1

# =========================
# ERROR ANALYSIS
# =========================
def explain_mistake(text, model_pred, confidence):
    rule_pred = heuristic_label(text)

    if confidence > 0.85:
        return None

    if rule_pred == model_pred:
        return None

    reasons = []

    if model_pred == 0 and rule_pred == 1:
        reasons = [
            "Model predicted FAKE but text is formal/factual",
            "Lack of emotional or clickbait language",
            "TF-IDF limitation (no context understanding)"
        ]

    elif model_pred == 1 and rule_pred == 0:
        reasons = [
            "Model predicted REAL but text contains emotional patterns",
            "Possible misleading vocabulary influence",
            "Bag-of-words ignores context"
        ]

    return rule_pred, reasons

# =========================
# UI HEADER
# =========================
st.title("📰 Fake News Detection Dashboard")
st.markdown("### AI-powered ML + DL Analysis System")

# =========================
# METRICS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Logistic Accuracy", f"{log_acc:.2%}")

with col2:
    st.metric("🧠 LSTM Accuracy", f"{lstm_acc:.2%}")

with col3:
    st.metric("⚖️ Best Model", "LSTM" if lstm_acc > log_acc else "Logistic")

st.divider()

# =========================
# INPUT
# =========================
text = st.text_area("✍️ Enter News Text", height=150)

# =========================
# ANALYZE
# =========================
if st.button("🚀 Analyze"):

    if text.strip() == "":
        st.warning("Please enter text")
        st.stop()

    vec = vectorizer.transform([clean_text(text)])

    # predictions
    log_pred = log_model.predict(vec)[0]
    log_prob = max(log_model.predict_proba(vec)[0])

    lstm_pred, lstm_prob = predict_lstm(text)

    hard_mode = hard_test_mode(text)

    # =========================
    # RESULTS
    # =========================
    st.subheader("📌 Prediction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.info("Logistic Regression")
        st.metric("Prediction", "REAL" if log_pred else "FAKE")
        st.progress(float(log_prob))

    with c2:
        st.success("LSTM Model")
        st.metric("Prediction", "REAL" if lstm_pred else "FAKE")
        st.progress(float(lstm_prob))

    st.divider()

    # =========================
    # CONFIDENCE COMPARISON
    # =========================
    st.subheader("📊 Model Confidence Comparison")

    df = pd.DataFrame({
        "Model": ["Logistic Regression", "LSTM"],
        "Confidence": [log_prob, lstm_prob]
    })

    fig = px.bar(df, x="Model", y="Confidence", color="Model", text="Confidence")
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # EXPLANATION
    # =========================
    st.subheader("🔍 Key Words Impact")

    scores = explain(text)

    for w, s in scores:
        if s > 0:
            st.write(f"🟢 **{w}** → Real signal ({s:.3f})")
        else:
            st.write(f"🔴 **{w}** → Fake signal ({s:.3f})")

    # =========================
    # HARD TEST MODE
    # =========================
    st.subheader("🧪 Hard Test Mode")

    if hard_mode:
        st.warning("Hard / Realistic News Detected")
        st.write("""
        ⚠️ This sample is challenging because:
        - Contains formal reporting language
        - No obvious clickbait signals
        - Requires context understanding
        """)
    else:
        st.success("Normal difficulty sample")

    # =========================
    # ERROR ANALYSIS
    # =========================
    st.subheader("⚠️ Why Model Might Be Wrong")

    rule_result = heuristic_label(text)
    mistake = explain_mistake(text, log_pred, log_prob)

    if mistake is None:
        st.success("Model prediction aligns with heuristic ✔")
    else:
        rule_label, reasons = mistake

        st.error("Potential Misclassification Detected")

        for r in reasons:
            st.write("• " + r)

        st.write("Rule-based label:",
                 "REAL ✅" if rule_label == 1 else "FAKE ❌")

    # =========================
    # FINAL INSIGHT
    # =========================
    st.subheader("⚖️ Final Insight")

    if lstm_acc > log_acc:
        st.success("LSTM performs better → understands context & sequence")
    else:
        st.info("Logistic Regression performs better → simpler & stable model")