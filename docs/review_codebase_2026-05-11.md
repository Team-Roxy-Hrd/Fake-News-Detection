```
══════════════════════════════════════════════════════════════
  CODE REVIEW REPORT
  Reviewed: Full Codebase (ui.py, notebooks/00–03, requirements.txt, README.md)
  Date: 2026-05-11
  Domain: AI/ML · DS · WEB (Streamlit)    Language: Python
══════════════════════════════════════════════════════════════
```

## Summary

```
Total issues found: 15
🔴 Critical: 0    🟠 High: 2    🟡 Medium: 7    🟢 Low: 3    🔵 Info: 3

Overall health: NEEDS WORK
```

The project has a clean, well-structured ML pipeline — data flows correctly from raw CSVs through cleaning, feature extraction, model training, and a Streamlit UI. The most pressing concerns are reproducibility gaps (no TF seed, empty `requirements.txt`) that prevent re-running or verifying results — critical for academic work. A recurring pattern across notebooks is the absence of standard ML safeguards: no early stopping, non-stratified validation split, and no cross-validation. The UI (`ui.py`) is in good shape after recent fixes; most remaining issues live in the notebooks and project configuration.

---

## Issues — Ranked by Severity

---

### 🟠 HIGH — Issue #1: `requirements.txt` Is Empty

**Location:** `requirements.txt`
**Dimension:** Reproducibility / AI-ML

**Problem:**
The file exists but has no content. Anyone trying to reproduce the project must manually guess all dependencies (`tensorflow`, `scikit-learn`, `streamlit`, `pandas`, `numpy`, `joblib`, `nltk`, `seaborn`, `matplotlib`) and their compatible versions. For academic submission this means the grader cannot reproduce results without significant guesswork.

**Example of the problem:**
```
# requirements.txt is completely empty
```

**Best Fix:**
Generate from the current environment and pin major versions. Exact pins (`==`) are best for reproducibility; compatible-release (`~=`) is acceptable for flexibility.

```
# requirements.txt
pandas~=2.0
numpy~=1.24
scikit-learn~=1.3
tensorflow~=2.13
streamlit~=1.28
joblib~=1.3
nltk~=3.8
matplotlib~=3.7
seaborn~=0.12
```

Generate automatically: `pip freeze > requirements.txt` then prune to direct dependencies only.

---

### 🟠 HIGH — Issue #2: No TensorFlow Random Seed — Non-Reproducible LSTM Training

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-7 (model build), cell-9 (training)
**Dimension:** AI-ML / Reproducibility

**Problem:**
`train_test_split` correctly uses `random_state=42`, but no global TF or NumPy seed is set before model construction and training. Keras weight initialization and dropout are both stochastic. Two runs of this notebook will produce different accuracy numbers, making academic comparison of results unreliable.

**Example of the problem:**
```python
# No seed set anywhere before this
model = Sequential([
    Embedding(...),
    LSTM(128, dropout=0.3, recurrent_dropout=0.3),
    ...
])
model.fit(X_train_pad, y_train, epochs=3, ...)
```

**Best Fix:**
Set all seeds at the top of the notebook, before any imports that use randomness.

```python
import random, os, numpy as np, tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

---

### 🟡 MEDIUM — Issue #3: No `EarlyStopping` — Model Undertrained

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-9
**Dimension:** AI-ML

**Problem:**
Training runs for exactly 3 epochs. The validation accuracy at epoch 3 is 95.07% and still climbing (epoch 1: 89.94%, epoch 2: 93.91%, epoch 3: 95.07%). The model has not converged. More epochs would improve accuracy, but training blindly risks overfitting. `EarlyStopping` solves both problems automatically.

**Example of the problem:**
```python
history = model.fit(
    X_train_pad, y_train,
    epochs=3,           # arbitrary, still improving
    batch_size=64,
    validation_split=0.1
)
```

**Best Fix:**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=20,          # high ceiling; early stopping exits when val_loss stops improving
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)
```

---

### 🟡 MEDIUM — Issue #4: EDA Word Frequency Analysis Not Filtered for Stopwords

**Location:** `notebooks/00_eda.ipynb` — cell-13, cell-15, cell-17
**Dimension:** Correctness / DS

**Problem:**
The most-common-words analysis shows "the", "and", "that", "for", "with" dominating both Fake and Real categories. These stopwords carry no discriminative signal. The plots and tables produced by this EDA are effectively useless because they show the same structure in both classes, obscuring the actual linguistic differences the project is trying to highlight.

**Example of the problem:**
```python
fake_words = re.findall(r"\b[a-z]{3,}\b", fake_words)
Counter(fake_words).most_common(20)
# Top result: ('the', 793032) — zero discriminative value
```

**Best Fix:**
```python
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))

fake_words = [w for w in re.findall(r"\b[a-z]{3,}\b", fake_text) if w not in stop]
Counter(fake_words).most_common(20)
# Now shows: trump, said, people, president, ... — actually meaningful
```

---

### 🟡 MEDIUM — Issue #5: `word_weight` Dict Rebuilt on Every `predict_lr()` Call

**Location:** `ui.py` — `predict_lr()`, line 65
**Dimension:** Performance

**Problem:**
Every time the user clicks "Analyze", `predict_lr()` rebuilds a 5000-entry dictionary from `tfidf.get_feature_names_out()` and `log_model.coef_[0]`. Both are static — they never change after models are loaded. This is pure wasted work on every button press.

**Example of the problem:**
```python
def predict_lr(text):
    vec = tfidf.transform([text])
    proba = log_model.predict_proba(vec)[0]
    word_weight = dict(zip(tfidf.get_feature_names_out(), log_model.coef_[0]))  # rebuilt every call
    ...
```

**Best Fix:**
Build it once at module load, alongside the models:

```python
log_model, tfidf, lstm_model, tokenizer = load_models()
WORD_WEIGHT = dict(zip(tfidf.get_feature_names_out(), log_model.coef_[0]))

def predict_lr(text):
    vec = tfidf.transform([text])
    proba = log_model.predict_proba(vec)[0]
    reasons = sorted(
        [(w, WORD_WEIGHT[w]) for w in text.lower().split() if w in WORD_WEIGHT],
        key=lambda x: abs(x[1]), reverse=True
    )[:8]
    ...
```

---

### 🟡 MEDIUM — Issue #6: LSTM Model Saved in Deprecated HDF5 Format

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-19
**Dimension:** Maintainability / AI-ML

**Problem:**
`model.save("lstm_model.h5")` triggers a Keras deprecation warning: *"This file format is considered legacy."* The `.h5` format does not support all Keras 3 features. The native `.keras` format is safer, smaller, and forward-compatible.

**Example of the problem:**
```python
model.save("../outputs/models/lstm_model.h5")
# WARNING: You are saving your model as an HDF5 file...
# This file format is considered legacy.
```

**Best Fix:**
```python
model.save("../outputs/models/lstm_model.keras")
```

And in `ui.py`, update the load call:
```python
lstm_model = load_model("outputs/models/lstm_model.keras")
```

---

### 🟡 MEDIUM — Issue #7: Single Train/Test Split — No Cross-Validation

**Location:** `notebooks/02_LG_TF-IDF.ipynb` — cell-3; `notebooks/03_LSTM_GLOVE.ipynb` — cell-1
**Dimension:** AI-ML

**Problem:**
Both models are evaluated on one fixed 80/20 split. The reported metrics (e.g., LR F1: 0.9394) are a single-sample estimate with unknown variance. A different `random_state` could produce a noticeably different number. For an academic comparison between two models, this means the difference in accuracy (LR: 94.60% vs LSTM: 94.52%) is not statistically meaningful — the models may perform identically on average.

**Best Fix:**
For LR (fast to train), use k-fold cross-validation:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

X_vec = tfidf.fit_transform(X)   # fit on all data for CV — or use Pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_vec, y, cv=cv, scoring="f1")
print(f"F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

For LSTM (expensive), at minimum try 3 different random seeds and report mean ± std.

---

### 🟡 MEDIUM — Issue #8: Keras `validation_split` Is Not Stratified

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-9
**Dimension:** AI-ML / Correctness

**Problem:**
`validation_split=0.1` takes the **last** 10% of the training array as the validation set, without shuffling or stratifying. If the data retains any ordering from the original concat (`fake_df` then `real_df`), this 10% slice could be heavily skewed toward one class, making validation loss a poor guide for early stopping and model selection.

**Example of the problem:**
```python
model.fit(X_train_pad, y_train,
          epochs=3,
          validation_split=0.1)   # last 10% of X_train_pad — no stratification
```

**Best Fix:**
Carve out a stratified validation set manually before training:

```python
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_pad, y_train, test_size=0.1, random_state=42, stratify=y_train
)

model.fit(X_tr, y_tr, epochs=20, batch_size=64,
          validation_data=(X_val, y_val),
          callbacks=[early_stop])
```

---

### 🟡 MEDIUM — Issue #9: `README.md` Is Empty

**Location:** `README.md`
**Dimension:** Maintainability

**Problem:**
The README contains no content. A grader or collaborator has no way to know: what the project does, how to install dependencies, how to run the notebooks in order, how to launch the UI, or where the raw data should be placed.

**Best Fix:**
Minimal README structure:

```markdown
# Fake News Detection

Compares Logistic Regression (TF-IDF) vs LSTM (GloVe) for fake news classification.

## Setup
pip install -r requirements.txt

## Run order
1. Place Fake.csv / True.csv in data/raw/
2. Place glove.6B.100d.txt in data/glove/
3. Run notebooks in order: 00 → 01 → 02 → 03
4. streamlit run ui.py
```

---

### 🟢 LOW — Issue #10: EDA Notebook Doesn't Drop Duplicates Before Analysis

**Location:** `notebooks/00_eda.ipynb` — cell-3
**Dimension:** Correctness / DS

**Problem:**
`notebooks/01_preprocessing.ipynb` calls `df.drop_duplicates()` but `00_eda.ipynb` does not. Word frequency counts in the EDA are therefore slightly inflated by duplicate articles. Minor for exploration, but creates an inconsistency between the EDA findings and the actual cleaned dataset.

**Best Fix:**
```python
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.drop_duplicates()   # add this line
```

---

### 🟢 LOW — Issue #11: Tokenizer Saved with `pickle` but Loaded with `joblib`

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` cell-19 (save) vs `ui.py` line 34 (load)
**Dimension:** Maintainability

**Problem:**
The tokenizer is saved with `pickle.dump()` in the notebook but loaded with `joblib.load()` in the UI. This works because joblib falls back to pickle for non-numpy objects, but it's a hidden dependency on implementation detail. It also means `import pickle` in the notebook and `import joblib` in the UI for what is conceptually the same operation.

**Best Fix:**
Use `joblib` consistently in both places:

```python
# In notebook 03:
import joblib
joblib.dump(tokenizer, "../outputs/models/tokenizer.pkl")

# In ui.py (already correct):
tokenizer = joblib.load("outputs/models/tokenizer.pkl")
```

---

### 🟢 LOW — Issue #12: No Input Length Validation in UI

**Location:** `ui.py` — `predict_lstm()`, line 73
**Dimension:** Reliability

**Problem:**
A user who pastes a very long document (e.g., 10,000 words) triggers LSTM padding to 300 tokens (truncation happens silently), but first the tokenizer processes all 10,000 words. There is no feedback that the input was truncated, which could confuse users who input a long article and get a result based only on its first 300 tokens.

**Best Fix:**
```python
text = st.text_area("Enter News Article", value=st.session_state.get("text", ""), max_chars=5000)

# In predict_lstm, add a note if truncated:
word_count = len(text.split())
if word_count > MAXLEN:
    st.caption(f"Note: input truncated from {word_count} to {MAXLEN} tokens for LSTM.")
```

---

### 🔵 INFO — Issue #13: TF-IDF Has No `min_df` / `max_df` Filtering

**Location:** `notebooks/02_LG_TF-IDF.ipynb` — cell-3
**Dimension:** AI-ML

**Problem:**
`TfidfVectorizer(max_features=5000, ngram_range=(1,2))` keeps the top 5000 terms by raw frequency. This can include very rare terms (appearing once or twice) that act as noise, and very common cross-class terms that carry no signal. Adding `min_df` and `max_df` is a standard practice.

**Suggestion:**
```python
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5, max_df=0.95)
```

---

### 🔵 INFO — Issue #14: `recurrent_dropout` Disables CuDNN Kernel

**Location:** `notebooks/03_LSTM_GLOVE.ipynb` — cell-7
**Dimension:** Performance / AI-ML

**Problem:**
`LSTM(128, dropout=0.3, recurrent_dropout=0.3)` — using `recurrent_dropout` prevents Keras from using the fast CuDNN LSTM implementation. The training log already shows "GPU will not be used on native Windows," so this does not affect this run, but it would significantly slow training if the environment is later migrated to Linux with a GPU.

**Suggestion:**
If GPU performance becomes important, move to Linux/WSL2 and use `dropout` only (not `recurrent_dropout`), or apply dropout as a separate `Dropout` layer after the LSTM output.

---

### 🔵 INFO — Issue #15: EDA and Preprocessing Both Re-load Raw CSVs

**Location:** `notebooks/00_eda.ipynb` cell-3; `notebooks/01_preprocessing.ipynb` cell-3
**Dimension:** Maintainability / DS

**Problem:**
Both notebooks independently read `Fake.csv` and `True.csv` and reconstruct the same base DataFrame. This is a normal pattern for Jupyter notebooks (each notebook is self-contained), but it means any change to how raw data is assembled must be made in two places.

**Suggestion:**
Acceptable as-is for a notebook-based project. If the project grows, consider a shared `src/data_loader.py` module imported by both notebooks.

---

## What's Done Well

- **Correct TF-IDF pipeline discipline** — `tfidf.fit_transform(X_train)` and `tfidf.transform(X_test)` are used correctly, with no data leakage from the test set into the vectorizer.
- **Correct tokenizer discipline** — `tokenizer.fit_on_texts(X_train)` only, then `texts_to_sequences` on test separately. Train/test split is performed before tokenization.
- **`stratify=y` in `train_test_split`** — both notebooks use stratified splitting, ensuring class balance is preserved in train and test sets.
- **Streamlit caching applied correctly** — `@st.cache_resource` for models, `@st.cache_data` for data and CM predictions, preventing expensive reloads on every interaction.
- **`show_result` correctly differentiates Fake vs Real** — uses `st.error` (red) for Fake and `st.success` (green) for Real rather than always showing green.

---

## Refactor Roadmap (Priority Order)

```
P0 — Fix before submission:
  [ ] Fill requirements.txt with all dependencies and versions
  [ ] Add tf.random.set_seed(42) + np.random.seed(42) in notebook 03

P1 — Fix this sprint:
  [ ] Add EarlyStopping callback in LSTM training (notebook 03)
  [ ] Filter stopwords in EDA word frequency analysis (notebook 00)
  [ ] Move word_weight dict construction out of predict_lr() to module level (ui.py)
  [ ] Save LSTM model as .keras format instead of .h5 (notebook 03)
  [ ] Manually create stratified validation split instead of validation_split=0.1 (notebook 03)
  [ ] Add cross-validation or multi-seed evaluation for LR model (notebook 02)
  [ ] Write a minimal README.md with setup and run instructions

P2 — Fix next time you touch this code:
  [ ] Drop duplicates before EDA word analysis (notebook 00)
  [ ] Standardize tokenizer save/load to joblib in both notebook and ui.py
  [ ] Add input length feedback in UI when text is truncated at MAXLEN tokens
```

---

## Domain-Specific Checklist

### AI / ML
- [x] Train/val/test split done before any preprocessing (both notebooks correct)
- [ ] Random seeds set for reproducibility — missing `tf.random.set_seed()` in notebook 03
- [x] Evaluation metric matches business objective — F1 + classification report used
- [x] Model artifacts versioned and saved — joblib + keras save used
- [ ] Validation split is stratified — `validation_split=0.1` is not stratified
- [ ] Cross-validation used or multiple seeds reported

### Performance
- [x] Expensive model loads cached with `@st.cache_resource`
- [x] Full-dataset CM predictions cached with `@st.cache_data`
- [ ] `word_weight` dict computed once, not per-call

### Reproducibility
- [ ] `requirements.txt` populated
- [ ] TF seed set before model construction
- [ ] README explains run order and data setup

```
══════════════════════════════════════════════════════════════
```
