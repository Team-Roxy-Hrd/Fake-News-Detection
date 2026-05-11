import streamlit as st
import pandas as pd
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MAXLEN = 300

st.set_page_config(
    page_title="Fake News AI Detector",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
# 🧠 Fake News AI Detection System
### Compare Machine Learning (TF-IDF) vs Deep Learning (LSTM)
---
""")


@st.cache_resource
def load_models():
    try:
        log_model = joblib.load("outputs/models/logistic_model.pkl")
        tfidf = joblib.load("outputs/models/tfidf_vectorizer.pkl")
        lstm_model = load_model("outputs/models/lstm_model.h5")
        tokenizer = joblib.load("outputs/models/tokenizer.pkl")
        return log_model, tfidf, lstm_model, tokenizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Run training first.")
        st.stop()


@st.cache_data
def load_stats():
    try:
        return pd.read_csv("data/processed/cleaned_fake_news.csv")
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}.")
        st.stop()


log_model, tfidf, lstm_model, tokenizer = load_models()
df_stats = load_stats()

total = len(df_stats)
fake_count = df_stats[df_stats["label"] == 1].shape[0]
real_count = df_stats[df_stats["label"] == 0].shape[0]
fake_ratio = (fake_count / total) * 100
real_ratio = (real_count / total) * 100


def predict_lr(text):
    vec = tfidf.transform([text])
    proba = log_model.predict_proba(vec)[0]
    label = "Fake" if proba.argmax() == 1 else "Real"
    conf = proba.max() * 100
    word_weight = dict(zip(tfidf.get_feature_names_out(), log_model.coef_[0]))
    reasons = sorted(
        [(w, word_weight[w]) for w in text.lower().split() if w in word_weight],
        key=lambda x: abs(x[1]), reverse=True
    )[:8]
    return label, conf, reasons


def predict_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    proba = lstm_model.predict(padded)[0][0]
    label = "Fake" if proba > 0.5 else "Real"
    conf = proba * 100 if label == "Fake" else (1 - proba) * 100
    return label, conf


def show_result(label, conf):
    msg = f"{label} ({conf:.2f}%)"
    if label == "Fake":
        st.error(msg)
    else:
        st.success(msg)


@st.cache_data
def get_cm_predictions(texts, labels):
    X_lr = tfidf.transform(texts)
    lr_preds = log_model.predict(X_lr)
    X_seq = tokenizer.texts_to_sequences(texts)
    X_pad = pad_sequences(X_seq, maxlen=MAXLEN)
    lstm_preds = (lstm_model.predict(X_pad) > 0.5).astype(int).flatten()
    return lr_preds, lstm_preds


def plot_cm(preds, y, title):
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


st.subheader("📊 Dataset Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Articles", total)
with col2:
    st.metric("Fake News", f"{fake_count} ({fake_ratio:.2f}%)")
with col3:
    st.metric("Real News", f"{real_count} ({real_ratio:.2f}%)")

st.markdown("---")

st.subheader("🧪 Try Sample News")
samples = {
    "Fake News Example": "Breaking: You won't believe what happened next! Scientists shocked by unbelievable discovery.",
    "Real News Example": "The government announced a new economic policy aimed at reducing inflation over the next fiscal year.",
    "Neutral News Example": "The president met officials in Washington to discuss international trade agreements."
}
choice = st.selectbox("Select Sample", list(samples.keys()))
if st.button("Load Sample"):
    st.session_state["text"] = samples[choice]

text = st.text_area("Enter News Article", value=st.session_state.get("text", ""))

if st.checkbox("📊 Show Confusion Matrix"):
    texts = df_stats["clean_text"].tolist()
    labels = df_stats["label"].values
    lr_preds, lstm_preds = get_cm_predictions(texts, labels)
    st.subheader("Logistic Regression Confusion Matrix")
    plot_cm(lr_preds, labels, "Logistic Regression")
    st.subheader("LSTM Confusion Matrix")
    plot_cm(lstm_preds, labels, "LSTM Model")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        lr_label, lr_conf, lr_reasons = predict_lr(text)
        lstm_label, lstm_conf = predict_lstm(text)

        st.subheader("🔍 Model Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 Logistic Regression (TF-IDF)")
            show_result(lr_label, lr_conf)
            st.progress(int(lr_conf))
            st.write("**Important Words:**")
            for w, score in lr_reasons:
                if score > 0:
                    st.write(f"🔴 {w} → FAKE influence ({score:.2f})")
                else:
                    st.write(f"🟢 {w} → REAL influence ({score:.2f})")

        with col2:
            st.markdown("### 🔵 LSTM (GloVe)")
            show_result(lstm_label, lstm_conf)
            st.progress(int(lstm_conf))
            st.caption("Word-level attribution is not available for this LSTM architecture.")

        st.markdown("---")

        if lr_label == lstm_label:
            st.success(f"✔ Both models agree: {lr_label}")
        else:
            st.warning("⚠ Models disagree — LSTM captures deeper context")
