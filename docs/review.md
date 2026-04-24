# Technical Review: Fake News Detection Project

**Reviewer:** Senior AI Engineer  
**Date:** 2026-04-24  
**Repository:** Fake-News-Detection  
**Branch:** main

---

## Part 1 — Technical Review (Independent of Requirements)

### 1.1 Project Structure

```
Fake-News-Detection/
├── docs/
│   └── project_reqs.md
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── outputs/
│   ├── models/
│   │   ├── log_model.pkl      (392 KB)
│   │   ├── lstm_model.pth     (5.9 MB)
│   │   ├── vectorizer.pkl     (2.0 MB)
│   │   └── vocab.pkl          (129 KB)
│   └── results/
│       ├── all_metrics.json
│       ├── dataset_stats.json
│       ├── log_metrics.json
│       └── lstm_metrics.json
├── report/
│   └── Fake_News_Detection_Report.pdf
├── README.md
├── requirements.txt
└── streamlit_app.py
```

**Assessment:** The structure is clean and logical. Separation of notebooks, outputs, and the app is sensible. However, the `data/` directory is absent from the repository — the raw dataset and processed CSV are not committed, which means the notebooks cannot be re-run by a fresh clone without manually placing the dataset.

---

### 1.2 Data & Exploration (`01_exploration.ipynb`)

- **Dataset:** Fake.csv + True.csv, merged into 44,898 samples.
- **Class balance:** Fake 23,481 (52.3%) / Real 21,417 (47.7%) — near-balanced, which is good.
- **Word frequency analysis:** Performed separately per class. Top words were dominated by common English tokens that are not informative without stopword removal (stopwords are correctly handled in preprocessing).
- **Output:** `dataset_stats.json` saved properly.

**Issues:**
- No null/missing value analysis documented in the notebook.
- No investigation of duplicate articles (news datasets frequently contain near-duplicates that cause inflated accuracy).
- No text length distribution analysis (useful for choosing LSTM max sequence length).
- Word frequency analysis is done before cleaning, reducing its analytical value.

---

### 1.3 Preprocessing (`02_preprocessing.ipynb`)

The `clean_text` function applies:
1. Lowercase conversion
2. Source artifact removal via regex (`reuters|21st century wire`)
3. Digit removal
4. Punctuation/special character removal
5. Whitespace tokenization
6. Stopword removal (NLTK English stopwords)
7. Short token filter (length < 2)

**Issues:**
- **Stopword library inconsistency:** The notebook uses NLTK stopwords, while `streamlit_app.py` uses `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`. These two lists are not identical, meaning preprocessing behavior differs between training and inference. This is a **data leakage / train-serve skew bug** — predictions in the app may be made on differently preprocessed text than what the models were trained on.
- **Digit removal is unnecessary and potentially harmful:** Removing all digits discards potentially discriminative numerical patterns (e.g., "2016 election", specific dates that signal context).
- **No lemmatization or stemming applied.** Not necessarily wrong, but worth noting as a potential improvement.
- **Feature combination:** Title + text concatenation into `content` is a reasonable strategy, though it dilutes title-specific signals.

---

### 1.4 Feature Extraction

**TF-IDF (Logistic Regression path):**
- `max_features=50,000`, `ngram_range=(1,2)` — well-configured. Bigrams capture multi-word expressions.
- Saved to `vectorizer.pkl` and loaded correctly in the app.

**Word Embeddings (LSTM path):**
- Custom learned embeddings (128-dim) built from the training vocabulary.
- `vocab_size`: 10,000 most frequent tokens, `min_freq=2`.
- Sequences capped at 200 tokens, zero-padded.
- **Note:** This is a learned embedding, not a pre-trained one (Word2Vec, GloVe, BERT). The requirement asks for an embedding method — this satisfies the spirit but uses no external pre-trained representations.

---

### 1.5 Model 1 — Logistic Regression

- **Input:** TF-IDF sparse matrix (50K features, bigrams).
- **Hyperparameters:** `max_iter=3000` — generous enough to ensure convergence.
- **Performance:**

| Metric    | Fake  | Real  | Macro |
|-----------|-------|-------|-------|
| Precision | 98.9% | 98.5% | 98.7% |
| Recall    | 98.7% | 98.7% | 98.7% |
| F1-score  | 98.8% | 98.6% | 98.7% |
| Accuracy  | **98.71%** | | |

- **Explainability:** Top-5 TF-IDF feature coefficients extracted per prediction — sound methodology.

**Issues:**
- No regularization strength (`C`) tuning reported. The default `C=1.0` is used without validation.
- No cross-validation performed — a single 80/20 split does not guarantee stability, especially given suspected near-duplicate articles in the dataset.
- 98.71% accuracy on this dataset is suspiciously high and consistent with known data leakage reports on the Kaggle Fake/True news dataset (source tags like "Reuters" remaining in text can trivially identify real news).

---

### 1.6 Model 2 — LSTM

**Architecture:**
```
Embedding(vocab_size, 128, padding_idx=0)
→ LSTM(128, 128, num_layers=2, dropout=0.3, batch_first=True)
→ Linear(128, 1)
→ BCEWithLogitsLoss
```

- **Training:** 5 epochs, Adam (lr=0.001), batch_size=32.
- **Sequence length:** 200 tokens (truncated/padded).
- **Performance:**

| Metric    | Fake  | Real  | Macro |
|-----------|-------|-------|-------|
| Precision | 94.2% | 98.9% | 96.6% |
| Recall    | 99.1% | 93.2% | 96.2% |
| F1-score  | 96.6% | 96.0% | 96.3% |
| Accuracy  | **96.31%** | | |

**Issues:**
- **No learning rate scheduling.** Adam with fixed lr=0.001 for 5 epochs may not converge optimally.
- **No early stopping.** Overfitting risk is unguarded, and 5 epochs is an arbitrary cutoff.
- **No validation loss curve plotted.** It is unknown whether the model overfits.
- **Bidirectional LSTM not used.** A BiLSTM typically outperforms unidirectional for text classification with no penalty.
- **`dropout=0.3` only applies between LSTM layers (num_layers=2), not before the final Linear layer.** An additional dropout before `fc` would be beneficial.
- The LSTM underperforms Logistic Regression by 2.4%, which is notable and likely caused by the data leakage issue (source tokens like "Reuters" dominate TF-IDF but are partially removed).

---

### 1.7 Streamlit Application (`streamlit_app.py`)

**Positive aspects:**
- Clean UI with side-by-side model comparisons.
- Confidence visualization with Plotly bar charts.
- Keyword importance display (TF-IDF coefficient × feature value).
- Error analysis via heuristic cross-validation.
- Hard test mode indicator.

**Issues:**

1. **Stopword mismatch (critical bug):** `clean_text` in `streamlit_app.py` uses `sklearn.ENGLISH_STOP_WORDS`, while notebooks use NLTK stopwords. Models were trained with NLTK stopwords, so inference uses a different vocabulary filter. This degrades prediction accuracy in production.

2. **LSTM probability interpretation is inverted for confidence display:**
   ```python
   st.progress(float(lstm_prob))
   ```
   `lstm_prob` is the sigmoid output toward class 1 (Real). If the model predicts Fake (prob < 0.5), the confidence bar shows a low value, not the actual confidence in the Fake prediction. This is misleading to the user.

3. **`log_prob = max(log_model.predict_proba(vec)[0])`** — taking the max of both class probabilities is correct for display but semantically imprecise. For example, if the model predicts Fake with 60% confidence, `max` returns 60%, but if it predicts Real with 60%, it also returns 60%. The display loses which class the confidence belongs to.

4. **Hard test mode is purely lexical and fragile.** The word "government" appearing once triggers "Hard / Realistic News Detected" regardless of context.

5. **Heuristic model tie-breaking defaults to Real (label=1)** when `fake_score == real_score`. This introduces a systematic bias against detecting fake news in ambiguous cases.

6. **No input length validation.** The app accepts and processes arbitrarily short or long texts without warning.

7. **Models loaded at global scope with no error handling.** If a model file is missing, the app crashes at startup with an unhelpful error.

8. **`streamlit` and `plotly` are missing from `requirements.txt`.** The app cannot be installed from the requirements file alone.

---

### 1.8 Requirements File

```
pandas
numpy
scikit-learn
nltk
torch
```

**Missing dependencies:**
- `streamlit` — required by `streamlit_app.py`
- `plotly` — required by `streamlit_app.py`
- `joblib` — imported explicitly (though bundled with scikit-learn, explicit declaration is best practice)

No version pins are specified. This is acceptable for a student project but risky in production — `torch` major versions break API compatibility regularly.

---

### 1.9 Code Quality Summary

| Area | Assessment |
|---|---|
| Project structure | Good |
| Notebook organization | Good — 3 clear stages |
| Preprocessing logic | Adequate, with stopword mismatch bug |
| TF-IDF configuration | Good |
| LR model | Good accuracy, lacks CV |
| LSTM architecture | Basic but functional |
| Explainability | Meaningful keyword extraction |
| Streamlit app | Well-designed UI, minor bugs |
| Requirements file | Incomplete |
| Data reproducibility | Broken — dataset not committed |

---

## Part 2 — Requirements Validation

### General Mandatory Requirements

#### REQ-G1: Apply text preprocessing techniques
**Status: PASS**

Implemented in `02_preprocessing.ipynb`:
- Lowercase conversion ✓
- Punctuation removal ✓
- Stopword removal (NLTK) ✓
- Tokenization (whitespace split) ✓

Minor gap: No lemmatization, but this is not required.

---

#### REQ-G2: Use two feature extraction methods — TF-IDF (required) + one embedding method
**Status: PARTIAL PASS**

- TF-IDF: Fully implemented with 50K features and bigrams ✓
- Embedding method: Custom learned LSTM embeddings (128-dim) ✓

**Gap:** The requirement lists "Word2Vec, GloVe, Transformer embeddings" as examples of embedding methods. The implemented approach uses randomly initialized embeddings trained end-to-end with the LSTM rather than a standalone, reusable embedding method. The embeddings are implicitly part of the LSTM rather than a distinct feature extraction stage. This is architecturally valid but does not demonstrate Word2Vec/GloVe/BERT usage specifically as a feature extraction step.

---

#### REQ-G3: Implement at least TWO models — one baseline + one advanced
**Status: PASS**

- Baseline model: Logistic Regression ✓
- Advanced model: LSTM (PyTorch, 2-layer, 128-hidden) ✓

---

#### REQ-G4: Perform a clear comparison between models
**Status: PASS**

Comparison documented in `03_modeling.ipynb` and in the Streamlit app:
- Accuracy comparison: LR 98.71% vs LSTM 96.31% ✓
- Strengths/weaknesses narrative: LR is simpler and more stable; LSTM has better fake recall (99.07%) but lower real recall ✓

---

#### REQ-G5: Provide evaluation metrics — Accuracy, Precision/Recall/F1, Confusion Matrix
**Status: PARTIAL PASS**

- Accuracy: Reported for both models ✓
- Precision / Recall / F1-score: Full classification reports saved as JSON ✓
- Confusion Matrix: **Not found in any notebook output or saved artifact.** The requirements explicitly require a confusion matrix. A confusion matrix plot or table is absent.

**Gap:** No confusion matrix visualization or printed matrix is present in the notebooks, the app, or the output files.

---

#### REQ-G6: Submit a final report
**Status: PASS**

`report/Fake_News_Detection_Report.pdf` (4.1 MB) is present ✓

---

### Project-Specific Requirements

#### REQ-P1: Build a system that classifies news articles as Fake or Real
**Status: PASS**

Implemented end-to-end: training pipeline in notebooks, inference in Streamlit app. Both LR and LSTM perform binary classification (0=Fake, 1=Real) ✓

---

#### REQ-P2: Understand patterns in fake vs real news
**Status: PASS**

- TF-IDF coefficient analysis reveals which n-grams are discriminative per class ✓
- Heuristic pattern detection in the app identifies exaggerated language (breaking, shocking, unbelievable) vs formal language (according to, announced, reported) ✓
- Dataset statistics saved (fake/real distribution) ✓

---

#### REQ-P3: Explain why a news article is classified as fake or real
**Status: PASS**

The `explain()` function in the app extracts up to 8 keywords with their signed TF-IDF impact scores:
- Negative score → Fake signal (red)
- Positive score → Real signal (green)

This matches the required output format: `Fake Reason: "you won't believe", "breaking"` ✓

---

#### REQ-P4: Extract important keywords (e.g., shocking, breaking)
**Status: PASS**

Implemented via:
1. TF-IDF coefficient extraction in `explain()` ✓
2. Heuristic keyword list (fake_signals, real_signals) in `heuristic_label()` ✓

---

#### REQ-P5: Identify common patterns across fake news
**Status: PASS**

Addressed through:
- TF-IDF top features per class (notebook analysis) ✓
- Heuristic patterns: fake news uses clickbait/emotional language; real news uses formal/factual language ✓

---

#### REQ-P6: Simple statistics (e.g., % of fake vs real news)
**Status: PASS**

`dataset_stats.json`:
- Fake: 23,481 (52.30%)
- Real: 21,417 (47.70%)

Displayed in `01_exploration.ipynb` ✓

---

#### REQ-P-TASK1: Preprocessing — lowercase, punctuation removal, stopwords, tokenization
**Status: PASS** (with stopword library inconsistency noted above)

---

#### REQ-P-TASK2: Feature Extraction — TF-IDF + Word embeddings
**Status: PARTIAL PASS** (see REQ-G2 above)

---

#### REQ-P-TASK3: Baseline Model — Logistic Regression
**Status: PASS** ✓

---

#### REQ-P-TASK4: Advanced Model — LSTM or Transformer
**Status: PASS** (LSTM implemented) ✓

---

#### REQ-P-TASK5: Evaluation — Accuracy, Confusion Matrix, Precision/Recall/F1, model comparison
**Status: PARTIAL PASS**

- Accuracy ✓
- Precision/Recall/F1 ✓
- Model comparison ✓
- **Confusion Matrix: MISSING** ✗

---

#### REQ-P-TASK6: Report — problem description, dataset, preprocessing, models, results, conclusion
**Status: PASS**

PDF report present at `report/Fake_News_Detection_Report.pdf` ✓ (content not independently verified beyond file presence)

---

## Part 3 — Summary

### Requirements Compliance

| Requirement | Status | Notes |
|---|---|---|
| Text preprocessing | **PASS** | All 4 steps implemented |
| TF-IDF feature extraction | **PASS** | 50K features, bigrams |
| Embedding feature extraction | **PARTIAL** | Learned LSTM embeddings, not pre-trained Word2Vec/GloVe/BERT |
| Baseline model (LR) | **PASS** | 98.71% accuracy |
| Advanced model (LSTM) | **PASS** | 96.31% accuracy |
| Model comparison | **PASS** | Accuracy + strengths/weaknesses |
| Accuracy metric | **PASS** | Both models |
| Precision/Recall/F1 | **PASS** | Full classification reports |
| **Confusion Matrix** | **FAIL** | Not present anywhere |
| Final report | **PASS** | PDF submitted |
| Classification (Fake/Real) | **PASS** | Two models, both work |
| Pattern understanding | **PASS** | TF-IDF + heuristics |
| Prediction explanation | **PASS** | Keyword impact scores |
| Important keywords | **PASS** | Extracted per prediction |
| Common patterns | **PASS** | Heuristic + statistical analysis |
| Simple statistics | **PASS** | Dataset distribution reported |

**Overall: 14/15 requirements met. 1 hard FAIL (Confusion Matrix). 2 partial passes (embedding type, confusion matrix also applies to evaluation task).**

---

### Critical Issues to Fix

1. **[CRITICAL] Confusion Matrix missing** — This is explicitly required. Add `sklearn.metrics.confusion_matrix` and plot it for both models in `03_modeling.ipynb`.

2. **[CRITICAL] Stopword library mismatch** — `streamlit_app.py` uses sklearn stopwords; notebooks use NLTK. Use the same list in both. Easiest fix: switch the app to use NLTK stopwords, matching training.

3. **[IMPORTANT] Missing `streamlit` and `plotly` in `requirements.txt`** — Add both.

4. **[IMPORTANT] LSTM confidence display misleads when predicting Fake** — Show `1 - lstm_prob` when the prediction is Fake.

5. **[MINOR] Possible data leakage** — Source tags (reuters) partially survived preprocessing. The 98.71% accuracy may be inflated. Consider a cleaned re-run with stronger source-artifact removal to verify true performance.

6. **[MINOR] No cross-validation** — A k-fold split would provide more robust performance estimates.

7. **[MINOR] Dataset not committed** — Include a download script or Kaggle API call so the project is reproducible.
