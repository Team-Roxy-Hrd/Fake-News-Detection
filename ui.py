import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# PAGE CONFIG
# ======================
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

# ======================
# LOAD MODELS
# ======================
log_model = joblib.load("outputs/models/logistic_model.pkl")
tfidf = joblib.load("outputs/models/tfidf_vectorizer.pkl")

lstm_model = load_model("outputs/models/lstm_model.h5")

with open("outputs/models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ======================
# LOAD DATA
# ======================
df_stats = pd.read_csv("data/processed/cleaned_fake_news.csv")

total = len(df_stats)
fake_count = df_stats[df_stats["label"] == 1].shape[0]
real_count = df_stats[df_stats["label"] == 0].shape[0]

fake_ratio = (fake_count / total) * 100
real_ratio = (real_count / total) * 100

# ======================
# EXPLAIN FUNCTIONS
# ======================
def explain_lr(text):
    vec = tfidf.transform([text])
    pred = log_model.predict(vec)[0]

    features = tfidf.get_feature_names_out()
    weights = log_model.coef_[0]

    word_weight = dict(zip(features, weights))

    words = text.lower().split()
    reasons = [(w, word_weight[w]) for w in words if w in word_weight]
    reasons = sorted(reasons, key=lambda x: abs(x[1]), reverse=True)[:8]

    label = "Fake" if pred == 1 else "Real"
    return label, reasons


def explain_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)

    pred = lstm_model.predict(padded)[0][0]
    label = "Fake" if pred > 0.5 else "Real"

    words = text.lower().split()
    important = [w for w in words if w in tokenizer.word_index]

    return label, important[:8]

# ======================
# CONFUSION MATRIX
# ======================
def plot_cm(model, X, y, title):
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ======================
# DATASET STATS
# ======================
st.subheader("📊 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Articles", total)

with col2:
    st.metric("Fake News", f"{fake_count} ({fake_ratio:.2f}%)")

with col3:
    st.metric("Real News", f"{real_count} ({real_ratio:.2f}%)")

st.markdown("---")

# ======================
# SAMPLE INPUTS
# ======================
st.subheader("🧪 Try Sample News")

samples = {
    "Fake News Example": "Breaking: You won’t believe what happened next! Scientists shocked by unbelievable discovery.",
    "Real News Example": "The government announced a new economic policy aimed at reducing inflation over the next fiscal year.",
    "Neutral News Example": "The president met officials in Washington to discuss international trade agreements."
}

choice = st.selectbox("Select Sample", list(samples.keys()))

if st.button("Load Sample"):
    st.session_state["text"] = samples[choice]

# ======================
# INPUT
# ======================
text = st.text_area(
    "Enter News Article",
    value=st.session_state.get("text", "")
)

# ======================
# CONFUSION MATRIX OPTION
# ======================
if st.checkbox("📊 Show Confusion Matrix"):

    y = df_stats["label"]

    st.subheader("Logistic Regression Confusion Matrix")
    X_lr = tfidf.transform(df_stats["clean_text"])
    plot_cm(log_model, X_lr, y, "Logistic Regression")

    st.subheader("LSTM Confusion Matrix")
    X_seq = tokenizer.texts_to_sequences(df_stats["clean_text"])
    X_pad = pad_sequences(X_seq, maxlen=300)
    plot_cm(lstm_model, X_pad, y, "LSTM Model")

# ======================
# PREDICTION
# ======================
if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter text")
    else:

        # ======================
        # LOGISTIC REGRESSION
        # ======================
        lr_vec = tfidf.transform([text])
        lr_proba = log_model.predict_proba(lr_vec)[0]

        lr_label = "Fake" if np.argmax(lr_proba) == 1 else "Real"
        lr_conf = np.max(lr_proba) * 100

        lr_label_exp, lr_reason = explain_lr(text)

        # ======================
        # LSTM
        # ======================
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=300)

        lstm_proba = lstm_model.predict(padded)[0][0]

        lstm_label = "Fake" if lstm_proba > 0.5 else "Real"
        lstm_conf = lstm_proba * 100 if lstm_label == "Fake" else (1 - lstm_proba) * 100

        lstm_label_exp, lstm_reason = explain_lstm(text)

        # ======================
        # OUTPUT
        # ======================
        st.subheader("🔍 Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 Logistic Regression (TF-IDF)")
            st.success(f"{lr_label} ({lr_conf:.2f}%)")
            st.progress(int(lr_conf))

            st.write("**Important Words:**")
            for w, score in lr_reason:
                if score > 0:
                    st.write(f"🔴 {w} → FAKE influence ({score:.2f})")
                else:
                    st.write(f"🟢 {w} → REAL influence ({score:.2f})")

        with col2:
            st.markdown("### 🔵 LSTM (GloVe)")
            st.success(f"{lstm_label} ({lstm_conf:.2f}%)")
            st.progress(int(lstm_conf))

            st.write("**Context Words:**")
            for w in lstm_reason:
                st.write(f"📌 {w}")

        st.markdown("---")

        # ======================
        # AGREEMENT
        # ======================
        if lr_label == lstm_label:
            st.success(f"✔ Both models agree: {lr_label}")
        else:
            st.warning("⚠ Models disagree — LSTM captures deeper context")