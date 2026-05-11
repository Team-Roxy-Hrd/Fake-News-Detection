import re
import json
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
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
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
            lstm = load_model("outputs/models/lstm_model.keras")
            with open("outputs/models/tokenizer.json", encoding="utf-8") as f:
                tok = tokenizer_from_json(json.load(f))
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
    "Fake example": (
        "SHOCK REPORT: Government Hiding Cure for Cancer to Protect Big Pharma Profits! "
        "Whistleblowers from inside the FDA have come forward with explosive documents proving that a natural cure for all "
        "forms of cancer has been suppressed for decades by corrupt government officials working hand-in-hand with pharmaceutical "
        "giants. The miracle treatment, derived from a common household plant, has a 100% success rate according to insider sources "
        "who risked their lives to expose the truth. Mainstream media refuses to cover this bombshell story because their corporate "
        "owners receive billions in advertising revenue from the same drug companies profiting off your suffering. Share this "
        "before it gets deleted! The globalist elite do not want you to see this. Doctors who have tried to publish findings were "
        "threatened, fired, and silenced. One brave researcher was found dead under mysterious circumstances just days after "
        "announcing a press conference. Wake up America — they are lying to you and your family members are dying because of it."
    ),
    "Real example": (
        "Federal Reserve raises interest rates by quarter point, signals possible pause in hiking cycle. "
        "The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday, bringing it "
        "to a range of 5.25 to 5.5 percent, the highest level in 22 years. Fed Chair Jerome Powell said policymakers would "
        "continue to make decisions meeting by meeting based on incoming economic data, leaving open the possibility that the "
        "central bank could hold rates steady at its next meeting in September. The move was widely expected by financial markets "
        "and marks the eleventh rate increase since March 2022, when the Fed began its most aggressive tightening campaign in "
        "four decades to bring down inflation that peaked above 9 percent last year. Inflation has since cooled to 3 percent "
        "in June, still above the Fed's 2 percent target. Powell acknowledged the progress but emphasized that officials need "
        "to see more evidence that price pressures are sustainably returning to target before considering rate cuts. "
        "The S&P 500 index rose modestly following the announcement as investors interpreted Powell's remarks as a sign "
        "the tightening cycle may be nearing its end."
    ),
    "Neutral example": (
        "Senate committee advances bipartisan infrastructure bill after weeks of negotiations. "
        "A bipartisan group of senators reached agreement on a roughly 1.2 trillion dollar infrastructure package on Tuesday, "
        "clearing a key procedural hurdle after weeks of tense negotiations over how to pay for roads, bridges, broadband "
        "internet and other public works projects. The bill passed out of committee by a vote of 69 to 30, with 19 Republicans "
        "joining all 50 Democrats in moving the legislation forward. Senate Majority Leader Chuck Schumer said he hoped to "
        "bring the bill to a full floor vote before the August recess. The package allocates 550 billion dollars in new federal "
        "spending on top of existing transportation funding, including 110 billion for roads and bridges, 73 billion to upgrade "
        "the power grid, 65 billion for broadband expansion and 39 billion to modernize public transit systems. President Biden, "
        "who helped broker the deal, called it a historic investment that would create millions of good-paying jobs. "
        "The White House acknowledged that a separate, larger social spending bill would still be needed to fulfill broader "
        "campaign promises on climate and social programs."
    ),
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
