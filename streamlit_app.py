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
import plotly.graph_objects as go
import os
import nltk
from nltk.corpus import stopwords

try:
    STOP_WORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))
MAX_SEQ_LEN = 200

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'reuters|21st century wire', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(tokens)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide",
    page_icon="📰",
)

# ── LSTM architecture (must match 03_modeling.ipynb exactly) ───────────────────
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 128, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))

# ── Load artefacts (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    log_model  = joblib.load("outputs/models/log_model.pkl")
    vectorizer = joblib.load("outputs/models/vectorizer.pkl")

    with open("outputs/models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Read vocab_size from the checkpoint itself so the model shape always
    # matches the saved weights, regardless of how vocab.pkl was built.
    checkpoint = torch.load("outputs/models/lstm_model.pth", map_location="cpu")

    vocab_size = checkpoint["vocab_size"]
    embed_dim  = checkpoint["embed_dim"]

    lstm = LSTMModel(vocab_size, embed_dim)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm.eval()

    with open("outputs/results/all_metrics.json") as f:
        metrics = json.load(f)

    dataset_stats = {}
    stats_path = "outputs/results/dataset_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            dataset_stats = json.load(f)

    return log_model, vectorizer, vocab, lstm, metrics, dataset_stats

log_model, vectorizer, vocab, lstm_model, metrics, dataset_stats = load_models()
log_m  = metrics["logistic_regression"]
lstm_m = metrics["lstm"]

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_logistic(text):
    vec  = vectorizer.transform([clean_text(text)])
    pred = int(log_model.predict(vec)[0])
    prob = float(log_model.predict_proba(vec)[0][pred])
    return pred, prob

def predict_lstm(text):
    words = clean_text(text).split()
    seq   = [vocab.get(w, 0) for w in words]
    seq   = seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq))
    tensor = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        logit = lstm_model(tensor).item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    return int(prob > 0.5), prob

def explain_logistic(text, top_n=10):
    """TF-IDF coefficient × term weight — positive = Real, negative = Fake."""
    vec   = vectorizer.transform([clean_text(text)])
    names = vectorizer.get_feature_names_out()
    coefs = log_model.coef_[0]
    idx   = vec.nonzero()[1]
    scores = [(names[i], float(coefs[i] * vec[0, i])) for i in idx]
    return sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:top_n]

def explain_lstm(text, top_n=10):
    """
    Gradient-based token saliency with direction.
    Sign: positive gradient dot embedding → pushes toward REAL (1).
    Negative → pushes toward FAKE (0).
    Magnitude: L2 norm of gradient (how much influence).
    Returns list of (word, signed_score).
    """
    words = clean_text(text).split()
    if not words:
        return []
    seq   = [vocab.get(w, 0) for w in words]
    seq   = seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq))
    tensor = torch.tensor([seq], dtype=torch.long)

    lstm_model.emb.weight.requires_grad_(True)
    emb_out = lstm_model.emb(tensor)
    emb_out.retain_grad()
    _, (h, _) = lstm_model.lstm(emb_out)
    logit = lstm_model.fc(lstm_model.dropout(h[-1]))
    lstm_model.zero_grad()
    logit.backward()

    if emb_out.grad is None:
        return []

    grad = emb_out.grad[0].detach().numpy()   # (seq_len, embed_dim)
    emb_vals = emb_out[0].detach().numpy()         # (seq_len, embed_dim)
    magnitude = np.linalg.norm(grad, axis=1)       # (seq_len,)
    direction = np.sign(np.sum(grad * emb_vals, axis=1))  # +1 real / -1 fake
    signed    = direction * magnitude              # signed saliency
    lstm_model.emb.weight.requires_grad_(False)

    n = min(len(words), MAX_SEQ_LEN)
    pairs = [(words[i], float(signed[i])) for i in range(n)]
    return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📰 Fake News Detection Dashboard")
st.markdown("#### NLP Course Project — Logistic Regression vs LSTM (GloVe embeddings)")
st.divider()

# ── Dataset statistics ─────────────────────────────────────────────────────────
if dataset_stats:
    st.subheader("🗂️ Dataset Statistics")
    ds1, ds2, ds3 = st.columns(3)
    ds1.metric("Total articles", f"{dataset_stats.get('total', 0):,}")
    ds2.metric("Fake articles",  f"{dataset_stats.get('fake', 0):,}  ({dataset_stats.get('fake_pct', 0):.1f}%)")
    ds3.metric("Real articles",  f"{dataset_stats.get('real', 0):,}  ({dataset_stats.get('real_pct', 0):.1f}%)")
    st.divider()

# ── Overall metrics ────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance (held-out test set)")

metric_keys  = ["accuracy", "precision", "recall", "f1"]
metric_names = ["Accuracy", "Precision", "Recall", "F1"]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Logistic Regression**")
    cols = st.columns(4)
    for col, name, key in zip(cols, metric_names, metric_keys):
        col.metric(name, f"{log_m[key]:.3f}")
with c2:
    st.markdown("**LSTM (GloVe)**")
    cols = st.columns(4)
    for col, name, key in zip(cols, metric_names, metric_keys):
        col.metric(name, f"{lstm_m[key]:.3f}")

# Confusion matrices
cm_log  = "outputs/results/log_confusion_matrix.png"
cm_lstm = "outputs/results/lstm_confusion_matrix.png"
if os.path.exists(cm_log) and os.path.exists(cm_lstm):
    st.markdown("**Confusion Matrices**")
    i1, i2 = st.columns(2)
    with i1:
        st.image(cm_log,  caption="Logistic Regression", use_container_width=True)
    with i2:
        st.image(cm_lstm, caption="LSTM",                use_container_width=True)

st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
st.subheader("✍️ Analyse a news article")
text_input = st.text_area("Paste news text here", height=160,
                           placeholder="Enter a headline or article body...")

if st.button("🚀 Analyse", type="primary"):

    if not text_input.strip():
        st.warning("Please enter some text first.")
        st.stop()

    log_pred,  log_prob  = predict_logistic(text_input)
    lstm_pred, lstm_prob = predict_lstm(text_input)

    # ── Predictions ────────────────────────────────────────────────────────────
    st.subheader("📌 Predictions")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Logistic Regression",
                  "🟢 REAL" if log_pred == 1 else "🔴 FAKE")
        st.progress(log_prob)
        st.caption(f"Confidence: {log_prob:.1%}")
    with r2:
        st.metric("LSTM (GloVe)",
                  "🟢 REAL" if lstm_pred == 1 else "🔴 FAKE")
        st.progress(lstm_prob)
        st.caption(f"Confidence: {lstm_prob:.1%}")

    if log_pred == lstm_pred:
        st.success("✅ Both models agree.")
    else:
        st.warning("⚠️ Models disagree — treat this result with caution.")

    # ── Required output format (per brief's example) ───────────────────────────
    # Best model prediction + top triggering keywords as a plain sentence
    best_pred  = lstm_pred  if lstm_m["f1"] > log_m["f1"] else log_pred
    best_label = "FAKE" if best_pred == 0 else "REAL"
    top_scores = explain_logistic(text_input, top_n=5)
    # fake keywords: negative score; real keywords: positive score
    fake_kws = [w for w, s in top_scores if s < 0][:3]
    real_kws  = [w for w, s in top_scores if s > 0][:3]
    trigger_kws = fake_kws if best_pred == 0 else real_kws
    reason_str  = (", ".join(f'"{w}"' for w in trigger_kws)
                   if trigger_kws else "no strong keyword signals found")

    if best_pred == 0:
        st.error(f"**Prediction: FAKE** — Reason: {reason_str}")
    else:
        st.success(f"**Prediction: REAL** — Reason: {reason_str}")

    st.divider()

    # ── Confidence chart ───────────────────────────────────────────────────────
    st.subheader("📊 Confidence comparison")
    fig = go.Figure(go.Bar(
        x=["Logistic Regression", "LSTM (GloVe)"],
        y=[log_prob, lstm_prob],
        marker_color=["#378ADD", "#1D9E75"],
        text=[f"{log_prob:.1%}", f"{lstm_prob:.1%}"],
        textposition="outside",
    ))
    fig.update_layout(yaxis=dict(range=[0, 1], title="Confidence"),
                      showlegend=False, height=300,
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Logistic Regression explanation ───────────────────────────────────────
    st.subheader("🔍 Key word influence — Logistic Regression")
    st.caption("TF-IDF weight × coefficient. Green = pushes toward REAL, red = pushes toward FAKE.")

    scores = explain_logistic(text_input)
    if scores:
        words, vals = zip(*scores)
        fig2 = go.Figure(go.Bar(
            x=list(vals), y=list(words),
            orientation="h",
            marker_color=["#1D9E75" if v > 0 else "#E24B4A" for v in vals],
        ))
        fig2.update_layout(
            xaxis_title="Influence score",
            yaxis=dict(autorange="reversed"),
            height=max(280, len(scores) * 32),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No recognisable features found in the vocabulary.")

    st.divider()

    # ── LSTM explanation ───────────────────────────────────────────────────────
    st.subheader("🧠 Token saliency — LSTM")
    st.caption("Green = pushes toward REAL, red = pushes toward FAKE. Bar length = strength of influence.")

    saliency = explain_lstm(text_input)
    if saliency:
        s_words, s_vals = zip(*saliency)
        fig3 = go.Figure(go.Bar(
            x=list(s_vals), y=list(s_words),
            orientation="h",
            marker_color=["#1D9E75" if v > 0 else "#E24B4A" for v in s_vals],
        ))
        fig3.update_layout(
            xaxis_title="Signed saliency (+ real / − fake)",
            yaxis=dict(autorange="reversed"),
            height=max(280, len(saliency) * 32),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough tokens to compute saliency.")

    st.divider()

    # ── Linguistic pattern check ───────────────────────────────────────────────
    st.subheader("🧪 Linguistic pattern check")
    FAKE_SIGNALS = ["breaking", "shocking", "unbelievable", "you won't believe",
                    "secret", "miracle", "urgent", "exposed", "bombshell"]
    REAL_SIGNALS = ["according to", "reported", "stated", "announced",
                    "data shows", "research", "study", "official"]

    text_lower  = text_input.lower()
    found_fake  = [w for w in FAKE_SIGNALS if w in text_lower]
    found_real  = [w for w in REAL_SIGNALS if w in text_lower]

    fc, rc = st.columns(2)
    with fc:
        st.markdown("**Sensationalist signals → Fake**")
        for w in found_fake:
            st.markdown(f"🔴 *{w}*")
        if not found_fake:
            st.markdown("None found")
    with rc:
        st.markdown("**Factual signals → Real**")
        for w in found_real:
            st.markdown(f"🟢 *{w}*")
        if not found_real:
            st.markdown("None found")

    st.divider()

    # ── Final verdict ──────────────────────────────────────────────────────────
    st.subheader("⚖️ Final Insight")
    if lstm_m["f1"] > log_m["f1"]:
        best, pred, prob = "LSTM (GloVe)", lstm_pred, lstm_prob
        st.success("LSTM performs better → understands context & word sequence.")
    else:
        best, pred, prob = "Logistic Regression", log_pred, log_prob
        st.info("Logistic Regression performs better → simpler & stable model.")

    label = "REAL ✅" if pred == 1 else "FAKE ❌"
    st.markdown(
        f"**{best}** classifies this article as **{label}** "
        f"with **{prob:.1%}** confidence."
    )