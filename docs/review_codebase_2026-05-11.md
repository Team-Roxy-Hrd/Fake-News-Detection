```
══════════════════════════════════════════════════════════════
  CODE REVIEW REPORT
  Reviewed: notebooks/00_eda.ipynb, 01_preprocessing.ipynb,
            02_LG_TF-IDF.ipynb, 03_LSTM_GLOVE.ipynb
  Date: 2026-05-11
  Domain: AI / DS (NLP, Machine Learning)
  Language: Python (Jupyter Notebooks)
══════════════════════════════════════════════════════════════
```
## Summary

  Total issues found: 12
  🔴 Critical:  2    🟠 High: 3    🟡 Medium: 3    🟢 Low: 2    🔵 Info: 2

  Overall health: NEEDS WORK

  The pipeline is structurally sound — data flows cleanly from raw CSVs
  through preprocessing to model training, TF-IDF is correctly fit only
  on training data, the tokenizer is also fit only on training data, and
  both models are evaluated with the right metrics. The two critical
  issues are in the explainability functions: both the Logistic Regression
  and LSTM "explain" functions return essentially all words in the input
  rather than the top contributing words, which directly fails the most
  distinctive graded requirement ("explain why classified as fake/real").
  Fixing those two functions is the only work required before submission.
  The remaining issues are best-practice gaps (reproducibility, early
  stopping, EDA quality) that are easy to address.

────────────────────────────────────────────────────────────
## Issues — Ranked by Severity
────────────────────────────────────────────────────────────

### 🔴 CRITICAL — Issue #1: LR Explainability Returns All Vocabulary Words, Not Top Contributors

**Location:** `notebooks/02_LG_TF-IDF.ipynb` — cell-7 (`explain` function)
**Dimension:** AI-ML / Correctness

**Problem:**
The `explain` function checks whether each raw input word exists anywhere in
`word_score` (a dict of all 5000 TF-IDF features). Because `word_score` is built
from all features, nearly every common word passes the filter, so the function
returns most of the input text as "important words." Additionally, `text.split()`
is not lowercased before comparison, so capitalised words like `"Breaking"` will
NOT match the lowercase TF-IDF feature `"breaking"`, making results inconsistent.
The project requirement explicitly requires showing which words DROVE the
classification decision — this function does not do that.

**Example of the problem:**
```python
word_score = dict(zip(features, weights))   # all 5000 features included

def explain(text):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    words = text.split()                    # not lowercased — case mismatch
    important_words = [w for w in words if w in word_score]  # any vocab word passes
    label = "Fake" if pred == 1 else "Real"
    return label, important_words
```

**Best Fix:**
Transform the text with TF-IDF, multiply non-zero feature values by the model
coefficient for that feature to get each word's signed contribution to the
prediction, then return only the top N by magnitude. This gives genuinely
informative reasons.

```python
def explain(text, top_n=10):
    cleaned = clean_text(text)              # reuse existing preprocessing
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "Fake" if pred == 1 else "Real"

    feature_indices = vec.nonzero()[1]
    contributions = [
        (features[i], weights[i] * vec[0, i])
        for i in feature_indices
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    important_words = [word for word, _ in contributions[:top_n]]

    return label, important_words
```

---

### 🔴 CRITICAL — Issue #2: LSTM Explainability Returns All Known Words, Not Prediction-Contributing Words

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-17 (`explain_text` function)
**Dimension:** AI-ML / Correctness

**Problem:**
The `explain_text` function returns every word from the input that exists in the
tokenizer's 20,000-word vocabulary. For a 50–100 word cleaned news snippet, almost
all words will be in that vocabulary, so the output is nearly the entire input text.
This satisfies neither the requirement ("important words or patterns that led to
classification") nor the example in the spec ("Reason: 'you won't believe',
'breaking'"). The function does not measure individual word contributions at all.

**Example of the problem:**
```python
def explain_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)
    pred = model.predict(padded)[0][0]
    label = "Fake" if pred > 0.5 else "Real"

    words = text.lower().split()
    # Returns EVERY word in the vocabulary — not the ones that matter
    important_words = [w for w in words if w in tokenizer.word_index]
    return label, important_words
```

**Best Fix:**
For an LSTM without attention layers, use input perturbation: blank out each word
one at a time, measure how much the prediction probability drops, and rank words by
their impact. This is simple, requires no extra libraries, and is genuinely
meaningful.

```python
def explain_text(text, top_n=10):
    words = text.lower().split()

    def predict_prob(word_list):
        seq = tokenizer.texts_to_sequences([" ".join(word_list)])
        padded = pad_sequences(seq, maxlen=300)
        return model.predict(padded, verbose=0)[0][0]

    base_prob = predict_prob(words)
    scores = []
    for i, word in enumerate(words):
        masked = words[:i] + ["<OOV>"] + words[i+1:]
        delta = base_prob - predict_prob(masked)
        scores.append((word, delta))

    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    label = "Fake" if base_prob > 0.5 else "Real"
    important_words = [w for w, _ in scores[:top_n]]
    return label, important_words
```

> Note: Perturbation adds one inference call per word. For short cleaned texts
> (~100 words) this takes a few seconds — acceptable for a demo. If speed matters,
> pre-filter to top-20 words by TF-IDF weight before perturbing.

---

### 🟠 HIGH — Issue #3: EDA Word Frequency Analysis Includes Stopwords — Insights Are Meaningless

**Location:** `notebooks/00_eda.ipynb` — cell-13 and cell-15
**Dimension:** AI-ML / Correctness

**Problem:**
The "most common words" analysis uses a simple `re.findall(r"\b[a-z]{3,}\b", ...)`
regex with no stopword removal. The top-20 words for both Fake and Real news are
therefore identical filler words: "the", "and", "that", "for", etc. This completely
defeats the purpose of the analysis — the project spec says "Fake news often uses
exaggerated words (e.g., 'shocking', 'unbelievable')" but the current output will
never surface those words because stopwords dominate.

**Example of the problem:**
```python
# Top 20 fake news words shown: 'the', 'and', 'that', 'for', 'with', 'trump' ...
fake_words = " ".join(df[df["label"]==1]["text"].astype(str)).lower()
fake_words = re.findall(r"\b[a-z]{3,}\b", fake_words)
Counter(fake_words).most_common(20)   # all stopwords at the top
```

**Best Fix:**
Apply the same stopword list used in preprocessing to make EDA findings match
what the models actually learn from.

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

fake_words = re.findall(r"\b[a-z]{3,}\b", fake_text_lower)
fake_words = [w for w in fake_words if w not in stop_words]
Counter(fake_words).most_common(20)
```

---

### 🟠 HIGH — Issue #4: No TensorFlow/Keras Random Seed — Results Not Reproducible

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-1 (imports section)
**Dimension:** AI-ML / Reproducibility

**Problem:**
The LSTM notebook sets `random_state=42` for `train_test_split` but never sets a
TensorFlow global seed. Keras weight initialisation and dropout masking use
TensorFlow's random state, so every run produces different weights and slightly
different accuracy numbers. This makes it impossible to reproduce the results you
report, and reviewers cannot verify your claims about LSTM performance.

**Example of the problem:**
```python
# random_state=42 only controls the data split, not model weights
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# model.fit(...) uses random init — different each run
```

**Best Fix:**
Add one line at the top of the imports cell:

```python
import tensorflow as tf
tf.random.set_seed(42)
```

---

### 🟠 HIGH — Issue #5: No Early Stopping — LSTM Is Still Improving at Final Epoch

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-9 (`model.fit`)
**Dimension:** AI-ML / Correctness

**Problem:**
The LSTM is trained for a hardcoded 3 epochs. The validation accuracy at epoch 3
is 0.9507, still trending upward from 0.8994 → 0.9391 → 0.9507, which means the
model has not converged. It is likely under-trained. With only 3 epochs, the LSTM
(0.9452 test accuracy) barely beats the much simpler Logistic Regression (0.9460),
which weakens the model comparison required by the project spec and makes the
"advanced model" look no better than the baseline.

**Example of the problem:**
```python
history = model.fit(
    X_train_pad, y_train,
    epochs=3,           # hardcoded, model still improving
    batch_size=64,
    validation_split=0.1
)
```

**Best Fix:**
Add `EarlyStopping` with `restore_best_weights=True` and increase `epochs` to
let the model train until it actually converges.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_pad, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)
```

---

### 🟡 MEDIUM — Issue #6: Keras Model Saved in Deprecated HDF5 (.h5) Format

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-19
**Dimension:** Reliability / Maintainability

**Problem:**
`model.save("../outputs/models/lstm_model.h5")` uses the legacy HDF5 format.
Keras raises this warning at runtime: "This file format is considered legacy. We
recommend using instead the native Keras format." The `.keras` format is more
robust and is the standard going forward.

**Example of the problem:**
```python
model.save("../outputs/models/lstm_model.h5")   # deprecated, raises warning
```

**Best Fix:**
```python
model.save("../outputs/models/lstm_model.keras")  # native format, no warning
```

---

### 🟡 MEDIUM — Issue #7: Tokenizer Saved with `pickle` — Fragile Across Versions

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-19
**Dimension:** Reliability

**Problem:**
`pickle.dump(tokenizer, f)` is version-sensitive — a tokenizer pickled with one
TensorFlow/Keras version may fail to load under another. Keras provides built-in
JSON serialisation for `Tokenizer` that is stable, human-readable, and the
recommended approach.

**Example of the problem:**
```python
import pickle
with open("../outputs/models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)   # version-sensitive, opaque binary
```

**Best Fix:**
```python
import json
tokenizer_json = tokenizer.to_json()
with open("../outputs/models/tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_json, f)

# To reload:
from tensorflow.keras.preprocessing.text import tokenizer_from_json
with open("../outputs/models/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(json.load(f))
```

---

### 🟡 MEDIUM — Issue #8: Class Imbalance Not Addressed in Either Model

**Location:** `notebooks/02_LG_TF-IDF.ipynb` — cell-3; `notebooks/03_LSTM_GLOVE.ipynb` — cell-7
**Dimension:** AI-ML

**Problem:**
The dataset has 34,618 Real vs 28,020 Fake articles (~55/45 split). Neither model
applies `class_weight`. Without it, both models may subtly bias toward the majority
class (Real), and the minority class (Fake) may have lower Recall than optimal.
For a project that must report per-class Precision/Recall/F1, this is worth
addressing.

**Best Fix:**
```python
# LR (02_LG_TF-IDF.ipynb)
model = LogisticRegression(max_iter=1000, class_weight="balanced")

# LSTM (03_LSTM_GLOVE.ipynb)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(..., class_weight=class_weight_dict)
```

---

### 🟢 LOW — Issue #9: `unique_word_ratio` Uses Non-Standard `+1` Denominator

**Location:** `notebooks/01_preprocessing.ipynb` — cell-9
**Dimension:** Correctness

**Problem:**
`unique_word_ratio` is computed as `len(set(words)) / (len(words) + 1)`. Since
empty texts are already filtered above this line, the `+1` is unnecessary and
introduces a small downward bias (a 1-word text gives 0.5 instead of 1.0).

**Example of the problem:**
```python
df["unique_word_ratio"] = df["clean_text"].apply(
    lambda x: len(set(x.split())) / (len(x.split()) + 1)  # biased for short texts
)
```

**Best Fix:**
```python
df["unique_word_ratio"] = df["clean_text"].apply(
    lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0.0
)
```

---

### 🟢 LOW — Issue #10: Logistic Regression Uses Single Split with Default Hyperparameters

**Location:** `notebooks/02_LG_TF-IDF.ipynb` — cell-3
**Dimension:** AI-ML / Reliability

**Problem:**
The LR model uses a single 80/20 split with default `C=1.0` and no validation of
whether that regularisation value is appropriate. A quick 5-fold CV check alongside
the existing evaluation would give more reliable numbers for the report's comparison
section.

**Best Fix:**
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    X_train_vec, y_train, cv=5, scoring="f1_macro"
)
print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

### 🔵 INFO — Issue #11: GloVe Embeddings Frozen (`trainable=False`) May Limit LSTM Performance

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-7
**Dimension:** AI-ML

**Problem:**
`trainable=False` means the GloVe vectors never update during training. Generic
GloVe embeddings may not perfectly capture fake-news domain vocabulary (political
jargon, clickbait phrasing). Allowing fine-tuning can improve accuracy.

**Suggestion:**
```python
Embedding(
    input_dim=max_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True          # allow fine-tuning on this domain
)
```

---

### 🔵 INFO — Issue #12: No Model Versioning — Retraining Silently Overwrites Artifacts

**Location:** All model notebooks
**Dimension:** AI-ML / Maintainability

**Problem:**
Models are saved as flat files with no metadata (training date, hyperparameters,
accuracy). If you retrain, the previous artifacts are silently overwritten, making
it hard to trace which numbers in the report came from which run.

**Suggestion:**
Save a sidecar JSON with each model artifact:

```python
import json, datetime

metadata = {
    "trained_at": datetime.datetime.now().isoformat(),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "params": {"max_features": 5000, "ngram_range": "(1,2)"}
}
with open("../outputs/models/logistic_model_meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

────────────────────────────────────────────────────────────
## What's Done Well
────────────────────────────────────────────────────────────

- **Correct TF-IDF fit/transform split:** `tfidf.fit_transform(X_train)` and
  `tfidf.transform(X_test)` — test data never touches the vectoriser fit.

- **Correct tokenizer fit:** `tokenizer.fit_on_texts(X_train)` only — the LSTM
  tokenizer is also correctly trained on training data alone, no leakage.

- **Stratified split throughout:** Both model notebooks use `stratify=y` in
  `train_test_split`, ensuring the class ratio is preserved in both partitions.

- **Solid preprocessing pipeline:** `clean_text` in `01_preprocessing.ipynb`
  covers lowercase, URL/email removal, punctuation stripping, number removal,
  stopword filtering, and lemmatisation — all project requirements are met.

- **Complete evaluation metrics:** Both models report Accuracy, Precision, Recall,
  F1-score per class, and Confusion Matrix — exactly what the spec requires.

- **Feature engineering included:** `word_count`, `char_count`, and
  `unique_word_ratio` are computed and visualised in the preprocessing notebook,
  showing genuine insight into structural differences between Fake and Real text.

────────────────────────────────────────────────────────────
## Refactor Roadmap (Priority Order)
────────────────────────────────────────────────────────────

  P0 — Fix before submission:
    [ ] Fix LR `explain` function to return top-N contributing words (Issue #1)
    [ ] Fix LSTM `explain_text` to use perturbation-based word importance (Issue #2)
    [ ] Remove stopwords from EDA word frequency analysis (Issue #3)
    [ ] Add `tf.random.set_seed(42)` to LSTM notebook (Issue #4)
    [ ] Add EarlyStopping + increase epochs in LSTM training (Issue #5)

  P1 — Fix this sprint:
    [ ] Switch Keras model save to `.keras` format (Issue #6)
    [ ] Replace pickle tokenizer save with `tokenizer.to_json()` (Issue #7)
    [ ] Add `class_weight="balanced"` to LR; compute class weights for LSTM (Issue #8)

  P2 — Fix next time you touch this code:
    [ ] Fix `unique_word_ratio` denominator formula (Issue #9)
    [ ] Add 5-fold CV to LR for more reliable comparison numbers (Issue #10)

────────────────────────────────────────────────────────────
## Domain-Specific Checklist
────────────────────────────────────────────────────────────

### AI / ML
  [x] Train/val/test split done before any fit-based preprocessing
  [ ] Random seeds set for reproducibility (missing tf.random.set_seed in LSTM)
  [x] Evaluation metric matches business objective (F1 + accuracy used)
  [ ] Model artifacts versioned and logged (flat files only, no metadata)
  [x] No data leakage: TF-IDF and tokenizer fit only on training data
  [ ] Explainability functions return genuinely important words (currently broken)

### Performance
  [x] No N+1 query patterns
  [x] Text preprocessing batched via pandas apply
  [ ] LSTM could be stronger with fine-tuned embeddings (trainable=True)

══════════════════════════════════════════════════════════════
```
