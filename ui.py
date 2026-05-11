import re
import numpy as np
import streamlit as st
import pandas as pd
import joblib

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MAXLEN = 300
TOP_N = 8

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")
st.title("Fake News Detection System")
st.caption("Logistic Regression (TF-IDF) vs LSTM (GloVe) — with word-level explanations")

# ── Resource loaders ────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        lr = joblib.load("outputs/models/logistic_model.pkl")
        tfidf = joblib.load("outputs/models/tfidf_vectorizer.pkl")
    except FileNotFoundError as e:
        st.error(f"Required model not found: {e}. Run the training notebooks first.")
        st.stop()

    lstm, tok = None, None
    if TF_AVAILABLE:
        try:
            lstm = load_model("outputs/models/lstm_model.h5")
            tok = joblib.load("outputs/models/tokenizer.pkl")
        except Exception:
            pass

    return lr, tfidf, lstm, tok


@st.cache_resource
def get_nlp_tools():
    if not NLTK_AVAILABLE:
        return set(), None
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        return set(stopwords.words("english")), WordNetLemmatizer()
    except Exception:
        return set(), None


@st.cache_data
def load_dataset_stats():
    try:
        df = pd.read_csv("data/processed/cleaned_fake_news.csv")
        total = len(df)
        fake = int((df["label"] == 1).sum())
        return total, fake, total - fake
    except FileNotFoundError:
        pass
    try:
        fake_df = pd.read_csv("data/raw/Fake.csv")
        real_df = pd.read_csv("data/raw/True.csv")
        total = len(fake_df) + len(real_df)
        return total, len(fake_df), len(real_df)
    except FileNotFoundError:
        return None, None, None

# ── Text preprocessing (matches notebook pipeline) ──────────────────────────

def clean_text(text: str, stop_words: set, lemmatizer) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ── Prediction + explanation ─────────────────────────────────────────────────

def predict_lr(text, lr, tfidf, stop_words, lemmatizer):
    cleaned = clean_text(text, stop_words, lemmatizer)
    vec = tfidf.transform([cleaned])
    proba = lr.predict_proba(vec)[0]
    label = "Fake" if proba.argmax() == 1 else "Real"
    confidence = proba.max() * 100

    features = tfidf.get_feature_names_out()
    weights = lr.coef_[0]
    col_indices = vec.nonzero()[1]
    arr = vec.toarray()[0]
    contributions = sorted(
        [(features[i], float(weights[i] * arr[i])) for i in col_indices],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return label, confidence, contributions[:TOP_N]


def predict_lstm(text, lstm, tokenizer, stop_words, lemmatizer):
    cleaned = clean_text(text, stop_words, lemmatizer)
    words = cleaned.split()
    if not words:
        return "Real", 50.0, []

    def get_prob(word_list):
        joined = " ".join(word_list).strip()
        if not joined:
            return 0.5
        seq = tokenizer.texts_to_sequences([joined])
        padded = pad_sequences(seq, maxlen=MAXLEN)
        return float(lstm.predict(padded, verbose=0)[0][0])

    base_prob = get_prob(words)
    label = "Fake" if base_prob > 0.5 else "Real"
    confidence = base_prob * 100 if label == "Fake" else (1 - base_prob) * 100

    # Perturbation: blank each word, rank by how much the prediction changes
    checked = words[:30]
    scores = []
    for i, word in enumerate(checked):
        masked = checked[:i] + [""] + checked[i + 1:]
        delta = base_prob - get_prob(masked)
        scores.append((word, delta))

    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return label, confidence, scores[:TOP_N]

# ── Rendering helpers ────────────────────────────────────────────────────────

def render_verdict(label: str, confidence: float):
    if label == "Fake":
        st.error(f"**FAKE** — {confidence:.1f}% confidence")
    else:
        st.success(f"**REAL** — {confidence:.1f}% confidence")
    st.progress(int(confidence))


def render_reasons(reasons: list):
    if not reasons:
        st.caption("No significant words matched the model's vocabulary.")
        return
    st.write("**Key words driving this prediction:**")
    for word, score in reasons:
        tag = "→ Fake" if score > 0 else "→ Real"
        st.write(f"- `{word}` {tag} ({score:+.3f})")

# ── Main app ─────────────────────────────────────────────────────────────────

lr, tfidf, lstm, tokenizer = load_models()
stop_words, lemmatizer = get_nlp_tools()

# Dataset statistics
total, fake_count, real_count = load_dataset_stats()
if total:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Articles", f"{total:,}")
    c2.metric("Fake News", f"{fake_count:,} ({fake_count / total:.1%})")
    c3.metric("Real News", f"{real_count:,} ({real_count / total:.1%})")
    st.markdown("---")

# Model availability notice
if not TF_AVAILABLE:
    st.info("TensorFlow not available (Python 3.13 not yet supported). Only Logistic Regression is active.")
elif lstm is None or tokenizer is None:
    st.info("LSTM model could not be loaded. Only Logistic Regression is active.")

# Sample picker
SAMPLES = {
    "Fake example": "BREAKING: You won't believe what they found! Scientists SHOCKED by unbelievable discovery mainstream media is hiding from you!",
    "Real example": "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, the eighth increase since March 2022.",
    "Neutral example": "The president met with cabinet officials on Tuesday to discuss the proposed infrastructure spending bill.",
}

if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

sample_choice = st.selectbox("Load a sample:", list(SAMPLES.keys()))
if st.button("Load sample"):
    st.session_state["article_text"] = SAMPLES[sample_choice]

article = st.text_area("Enter news article text:", key="article_text", height=160)

if st.button("Analyze", type="primary"):
    if not article.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.subheader("Results")
        lr_label, lr_conf, lr_reasons = predict_lr(article, lr, tfidf, stop_words, lemmatizer)

        lstm_ready = lstm is not None and tokenizer is not None

        if lstm_ready:
            with st.spinner("Running LSTM analysis (perturbation-based explanation may take a few seconds)…"):
                lstm_label, lstm_conf, lstm_reasons = predict_lstm(
                    article, lstm, tokenizer, stop_words, lemmatizer
                )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Logistic Regression (TF-IDF)")
                render_verdict(lr_label, lr_conf)
                render_reasons(lr_reasons)
            with col2:
                st.markdown("#### LSTM (GloVe)")
                render_verdict(lstm_label, lstm_conf)
                render_reasons(lstm_reasons)

            st.markdown("---")
            if lr_label == lstm_label:
                st.success(f"Both models agree: **{lr_label}**")
            else:
                st.warning(
                    f"Models disagree — LR says **{lr_label}**, LSTM says **{lstm_label}**. "
                    "LSTM captures sequential context that TF-IDF misses."
                )
        else:
            st.markdown("#### Logistic Regression (TF-IDF)")
            render_verdict(lr_label, lr_conf)
            render_reasons(lr_reasons)
