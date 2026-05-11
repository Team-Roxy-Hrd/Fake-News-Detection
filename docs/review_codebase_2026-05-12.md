```
══════════════════════════════════════════════════════════════
  CODE REVIEW REPORT
  Reviewed: notebooks/ (00_eda, 01_preprocessing, 02_LG_TF-IDF, 03_LSTM_GLOVE)
  Date: 2026-05-12
  Domain: DS, AI, ML      Language: Python (Jupyter Notebooks)
══════════════════════════════════════════════════════════════
```
## Requirements Coverage

| Requirement | Status | Notes |
|---|---|---|
| Text preprocessing (lowercase, punct, stopwords, tokenization) | ✅ Done | `01_preprocessing.ipynb` |
| TF-IDF feature extraction | ✅ Done | `02_LG_TF-IDF.ipynb` |
| Word embedding (GloVe) | ✅ Done | `03_LSTM_GLOVE.ipynb` |
| Baseline model — Logistic Regression | ✅ Done | 94.6% accuracy |
| Advanced model — LSTM | ✅ Done | 94.5% accuracy |
| Accuracy / Precision / Recall / F1 / Confusion Matrix | ✅ Done | Both notebooks |
| Classification with explanation (important words) | ✅ Done | Both notebooks |
| EDA: class distribution, word analysis | ✅ Done | `00_eda.ipynb` |
| Class imbalance handling | ✅ Done | `class_weight="balanced"` in both |
| Model artifacts saved with metadata JSON | ✅ Done | Both notebooks |
| Cross-validation | ✅ Done | 5-fold CV in LG notebook |
| Final written report | ❌ MISSING | Mandatory deliverable |
| Common fake-news patterns systematic analysis | ⚠️ Partial | Top words shown, not pattern-focused |
| Explicit cross-model comparison section | ⚠️ Partial | Results in separate notebooks only |

---

## Summary

  Total issues found: 11
  🔴 Critical: 0    🟠 High: 2    🟡 Medium: 4    🟢 Low: 3    🔵 Info: 2

  Overall health: NEEDS WORK

  The project correctly implements all core NLP pipeline stages — preprocessing,
  TF-IDF + GloVe feature extraction, Logistic Regression baseline, LSTM advanced
  model, per-prediction explainability, and model artifact saving. The major
  previous issues (broken explain functions, frozen embeddings, hardcoded epochs)
  have been resolved. What remains is one missing mandatory deliverable (the final
  written report), one high-severity notebook hygiene issue (stale outputs that
  contradict the current codebase), and several medium issues around presentation
  quality that affect the grade. There is no overengineering — the code is lean
  and appropriately scoped for an academic project.

────────────────────────────────────────────────────────────
## Issues — Ranked by Module and Severity
────────────────────────────────────────────────────────────

### 🟠 HIGH — Issue #1: Final Written Report Is Missing
**Module:** Project deliverables
**Dimension:** Requirements compliance

**Problem:**
The project requirements (General Requirements point 6, and Project 3 Tasks
point 6) explicitly mandate a final report. The report must include: problem
description, dataset description, preprocessing steps, models used, results
and comparison, and conclusion. No such document exists in `docs/` or anywhere
in the repository. `docs/fixes_2026-05-12.md` and `docs/review_codebase_2026-05-11.md`
are internal development artefacts, not a submission report. This is a graded
mandatory deliverable — its absence will directly cost marks.

**What it must include (per spec):**
```
- Problem description & Dataset description
- Preprocessing steps
- Models used
- Results and comparison (Accuracy, Strengths & Weaknesses)
- Conclusion
```

**Fix:**
Create `docs/report.md` (or a PDF) containing each required section.
The notebook outputs already contain all the numbers — this is a writing task.
Copy the key metrics from notebooks:
- LG: Accuracy 94.6%, F1-macro ~0.94, 5-fold CV F1: (run to get value)
- LSTM: Accuracy 94.5%, F1-macro ~0.94
Include the confusion matrix figures and the explainability examples.

---

### 🟠 HIGH — Issue #2: LSTM Stale Outputs Use ~62k Rows vs Preprocessed 44k Rows
**Module:** `notebooks/03_LSTM_GLOVE.ipynb`, `notebooks/01_preprocessing.ipynb`
**Dimension:** AI-ML / Reproducibility / Correctness

**Problem:**
`01_preprocessing.ipynb` saves a cleaned CSV with 44,058 rows (after
deduplication). But `03_LSTM_GLOVE.ipynb`'s stored output shows:
`Train size: 50106 / Test size: 12527` — total 62,633 rows, which matches
the raw dataset size before deduplication was applied. This means the LSTM
notebook was last executed against an older version of `cleaned_fake_news.csv`
that predates the dedup fix. The stored LSTM performance numbers therefore
do not reflect the current data pipeline, making the notebook's saved outputs
misleading and the cross-model comparison (LG vs LSTM) based on different data.

**Example of the problem:**
```
# 01_preprocessing.ipynb output
Shape: (44058, 5)   # after dedup

# 03_LSTM_GLOVE.ipynb stored output (stale)
Train size: 50106   # 50106 + 12527 = 62633 ≈ raw dataset, not the cleaned one
Test size:  12527
```

**Fix:**
Re-run `03_LSTM_GLOVE.ipynb` end-to-end after `01_preprocessing.ipynb` so
all stored outputs reflect the same 44,058-row deduplicated dataset. Then
both model notebooks will have been evaluated on identical data.

---

### 🟡 MEDIUM — Issue #3: Stale `ParserError: out of memory` in EDA Notebook
**Module:** `notebooks/00_eda.ipynb` — Cell 3
**Dimension:** Maintainability / Correctness

**Problem:**
Cell 3 of `00_eda.ipynb` has a stored `ParserError: Error tokenizing data.
C error: out of memory` error from a previous failed run. The cell code is
correct and the same CSV loads fine in `01_preprocessing.ipynb`. This stale
error output makes the notebook appear broken to anyone who opens it without
re-running it, and will confuse graders who review the saved notebook.

**Example of the problem:**
```
# Stored output in Cell 3 of 00_eda.ipynb
ParserError: Error tokenizing data. C error: out of memory
```

**Fix:**
Re-run `00_eda.ipynb` from top to bottom (Kernel → Restart & Run All) and
save it after a clean execution. The error will be replaced by the correct
dataframe output.

---

### 🟡 MEDIUM — Issue #4: Stale Word-Frequency Outputs Show Unfiltered Stopwords
**Module:** `notebooks/00_eda.ipynb` — Cells 13, 15
**Dimension:** AI-ML / Correctness (output vs code mismatch)

**Problem:**
The code in cells 13 and 15 correctly filters stopwords before counting word
frequencies. However, the stored cell outputs still show the old unfiltered
results — `('the', 793032)`, `('and', 341293)` etc. as the top words. Anyone
reading the notebook sees stopwords dominating the chart, which contradicts
the purpose of the analysis and the requirement to "identify common patterns
in fake vs real news." The fix was applied to the code but the notebook was
never re-run to update the outputs.

**Example of the problem:**
```python
# Code (correct — filters stopwords):
fake_words = [w for w in fake_words if w not in stop_words]

# But stored output (stale — unfiltered):
[('the', 793032), ('and', 341293), ('that', 212057), ...]
```

**Fix:**
Re-run `00_eda.ipynb` (same as Issue #3 fix — one re-run clears both issues).

---

### 🟡 MEDIUM — Issue #5: No Explicit Cross-Model Comparison Section
**Module:** All model notebooks
**Dimension:** Requirements compliance / Maintainability

**Problem:**
The requirement specifies: "Perform a clear comparison between models:
Accuracy (or relevant metric), Strengths & weaknesses." Currently the
comparison is implicit — a reader must look at LG results in `02_LG_TF-IDF.ipynb`
and LSTM results in `03_LSTM_GLOVE.ipynb` separately and mentally compare.
There is no dedicated cell or notebook that places both sets of numbers
side-by-side. This makes graders do extra work and risks losing marks on
the comparison criterion.

**Fix:**
Add a comparison cell at the end of `03_LSTM_GLOVE.ipynb` (or create a
`04_comparison.ipynb`) with an explicit summary table:

```python
import pandas as pd

comparison = pd.DataFrame({
    "Model":     ["Logistic Regression (TF-IDF)", "LSTM (GloVe)"],
    "Accuracy":  [0.9460, 0.9452],
    "F1-macro":  [0.9460, 0.9440],
    "Train time":["fast (~seconds)",              "slow (~20 min)"],
    "Explainability": ["TF-IDF × coef (fast)", "Perturbation (slow, O(n) calls)"],
})
print(comparison.to_markdown(index=False))

# Also write a short strengths/weaknesses commentary
```

---

### 🟡 MEDIUM — Issue #6: Fake News "Common Patterns" Analysis Is Superficial
**Module:** `notebooks/00_eda.ipynb`
**Dimension:** Requirements compliance / AI-ML

**Problem:**
The project specification explicitly requires: "The system should identify
patterns such as: Fake news often uses exaggerated words (e.g., 'shocking',
'unbelievable'); Real news uses more formal and factual language." The current
EDA only shows raw word frequencies per class, which are dominated by neutral
political terms (`trump`, `said`, `government`). There is no analysis
specifically targeting the discriminative vocabulary — words that appear
disproportionately in Fake vs Real — which is what the spec is asking for.

**Fix:**
Add a cell that computes log-likelihood ratio or simple frequency ratio for
each word between Fake and Real to surface the most discriminative terms:

```python
from collections import Counter

# Already have fake_words and real_words lists from earlier cells
fake_freq = Counter(fake_words)
real_freq = Counter(real_words)

total_fake = sum(fake_freq.values())
total_real = sum(real_freq.values())

discriminative = {}
for word in set(fake_freq) | set(real_freq):
    f_rate = fake_freq.get(word, 0) / total_fake
    r_rate = real_freq.get(word, 0) / total_real
    if r_rate > 0:
        discriminative[word] = f_rate / r_rate  # ratio > 1 = more Fake

# Top words skewed toward Fake
sorted_fake_biased = sorted(discriminative.items(), key=lambda x: -x[1])
print("Words most associated with Fake news:")
print(sorted_fake_biased[:20])
```

---

### 🟢 LOW — Issue #7: LSTM Metadata Missing F1 Metric (Inconsistent with LG)
**Module:** `notebooks/03_LSTM_GLOVE.ipynb` — Cell 19 (save block)
**Dimension:** Maintainability / Reliability

**Problem:**
The LG metadata JSON saves both `accuracy` and `f1_macro`. The LSTM metadata
JSON saves only `accuracy`. This inconsistency makes automated comparison
of the two metadata files unreliable, and means the LSTM log is less
informative for the report.

**Example of the problem:**
```python
# LG metadata (02_LG_TF-IDF.ipynb) — has f1_macro ✅
metadata = {"accuracy": ..., "f1_macro": ..., ...}

# LSTM metadata (03_LSTM_GLOVE.ipynb) — missing f1_macro ❌
metadata = {"accuracy": ..., "params": ...}
```

**Fix:**
```python
from sklearn.metrics import f1_score

metadata = {
    "trained_at": datetime.datetime.now().isoformat(),
    "accuracy": float(accuracy_score(y_test, y_pred_flat)),
    "f1_macro": float(f1_score(y_test, y_pred_flat, average="macro")),
    "params": {"max_words": 20000, "max_len": 300, "epochs": "early_stopped", "batch_size": 64}
}
```

---

### 🟢 LOW — Issue #8: `explain_text` in LSTM Does Not Pre-clean Input
**Module:** `notebooks/03_LSTM_GLOVE.ipynb` — Cell 17
**Dimension:** Correctness / Reliability

**Problem:**
The LSTM `explain_text` function splits on `.lower()` only, but the tokenizer
was trained on text that went through the full `clean_text()` pipeline (URL
removal, punctuation removal, number removal, lemmatization, stopword removal).
If the sample input contains punctuation or stopwords, those tokens will be
looked up in the tokenizer and mapped to `<OOV>`, producing slightly inaccurate
perturbation scores. For the demo example `"Breaking news: shocking discovery..."`
the colon and stopwords (`about`) are present.

**Fix:**
Apply the same cleaning function before splitting:

```python
from notebook_01 import clean_text  # or redefine inline

def explain_text(text, top_n=10):
    cleaned = clean_text(text)       # <-- pre-clean first
    words = cleaned.split()
    ...
```

Or at minimum, import/redefine `clean_text` in the LSTM notebook and apply it.

---

### 🟢 LOW — Issue #9: `explain_text` O(n) Model Inference Calls Per Word
**Module:** `notebooks/03_LSTM_GLOVE.ipynb` — Cell 17
**Dimension:** Performance

**Problem:**
The perturbation-based explainability calls `model.predict()` once per word in
the input — for a 500-word article that is 500 separate forward passes. On CPU
this is already very slow (each pass ~60ms in the stored output → ~30 seconds
for a typical article). While acceptable for an academic demo, the function will
hang noticeably on long texts and will not scale to multiple explanations.

**Fix (simple for academic context):**
Batch all masked variants into a single predict call:

```python
def explain_text(text, top_n=10):
    words = text.lower().split()
    seqs = []
    for i in range(len(words)):
        masked = words[:i] + ["<OOV>"] + words[i+1:]
        seqs.append(" ".join(masked))

    all_seqs = tokenizer.texts_to_sequences([" ".join(words)] + seqs)
    all_padded = pad_sequences(all_seqs, maxlen=300)
    probs = model.predict(all_padded, verbose=0).flatten()

    base_prob = probs[0]
    scores = [(words[i], base_prob - probs[i+1]) for i in range(len(words))]
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    label = "Fake" if base_prob > 0.5 else "Real"
    return label, [w for w, _ in scores[:top_n]]
```

This reduces 500 calls to 1 batched call.

---

### 🔵 INFO — Issue #10: `epochs` Field in LSTM Metadata Is a String, Not Int
**Module:** `notebooks/03_LSTM_GLOVE.ipynb` — Cell 19
**Dimension:** Maintainability

**Problem:**
The LSTM metadata stores `"epochs": "early_stopped"` — a string — while all
other numeric params are integers. If anything parses this JSON to compare
training configurations, it will fail or need special-casing. The actual epoch
count is available from `len(history.history["accuracy"])`.

**Fix:**
```python
metadata = {
    ...
    "params": {
        "max_words": 20000,
        "max_len": 300,
        "epochs_run": len(history.history["accuracy"]),   # actual int
        "epochs_max": 15,
        "batch_size": 64
    }
}
```

---

### 🔵 INFO — Issue #11: Plot Style Inconsistent Across Notebooks
**Module:** `notebooks/00_eda.ipynb`, `notebooks/01_preprocessing.ipynb`
**Dimension:** Maintainability / Readability

**Problem:**
`00_eda.ipynb` sets `plt.style.use("ggplot")` but the remaining three notebooks
use the default Matplotlib style. This creates visual inconsistency in a project
where all notebooks contribute to the same final submission. Minor, but affects
presentation quality.

**Fix:**
Add `plt.style.use("ggplot")` (or any consistent style) to the imports cell
of `01_preprocessing.ipynb`, `02_LG_TF-IDF.ipynb`, and `03_LSTM_GLOVE.ipynb`.

────────────────────────────────────────────────────────────
## What's Done Well
────────────────────────────────────────────────────────────

- **TF-IDF fit/transform discipline**: `tfidf.fit_transform(X_train)` then
  `tfidf.transform(X_test)` — no data leakage from test into the vectorizer.

- **Explainability is genuinely informative**: Both `explain()` (TF-IDF × coef)
  and `explain_text()` (perturbation-based) produce meaningful ranked word lists
  that directly address the project requirement for prediction reasoning.

- **Class imbalance handled in both models**: `class_weight="balanced"` in LG
  and `compute_class_weight` + `class_weight_dict` in LSTM ensures neither
  model ignores the minority class.

- **Model artifact management is solid**: Both models save `.pkl`/`.keras`,
  matching tokenizer/vectorizer, and a JSON metadata file. The LSTM uses
  `tokenizer.to_json()` instead of pickle — correct and version-stable.

- **EarlyStopping with `restore_best_weights=True`**: The LSTM won't overfit
  to the last epoch; it restores the checkpoint where val_accuracy peaked.
  Combined with `epochs=15`, this is the right setup for convergence without
  overfitting.

────────────────────────────────────────────────────────────
## Refactor Roadmap (Priority Order)
────────────────────────────────────────────────────────────

  P0 — Fix before submission:
    [ ] Write the final report (Issue #1) — mandatory graded deliverable
    [ ] Re-run 00_eda.ipynb (clears Issues #3 and #4 in one step)
    [ ] Re-run 03_LSTM_GLOVE.ipynb after preprocessing (Issue #2 — align data sizes)
    [ ] Add cross-model comparison cell/notebook (Issue #5)
    [ ] Add discriminative word analysis to EDA (Issue #6 — spec requirement)

  P1 — Fix this sprint (quality improvements):
    [ ] Add F1-macro to LSTM metadata (Issue #7)
    [ ] Pre-clean input in explain_text (Issue #8)

  P2 — Fix next time you touch this code:
    [ ] Batch explain_text model calls (Issue #9)
    [ ] Fix epochs metadata to int (Issue #10)
    [ ] Align plot styles across notebooks (Issue #11)

────────────────────────────────────────────────────────────
## Domain-Specific Checklist
────────────────────────────────────────────────────────────

### AI / ML
  [x] Train/val/test split done before any preprocessing (TF-IDF fit on train only)
  [x] Random seeds set for reproducibility (random_state=42, tf.random.set_seed(42))
  [x] Evaluation metric matches objective (F1-macro on balanced classes)
  [x] Model artifacts versioned and logged (JSON metadata saved)
  [ ] Re-run all notebooks after data pipeline changes to sync stored outputs
  [ ] Common patterns analysis surfacing discriminative vocabulary

### Performance
  [ ] explain_text batched to one model.predict() call (currently O(n) calls)
  [x] No N+1 patterns in training/evaluation paths
  [x] GloVe embeddings loaded once; embedding matrix built once

══════════════════════════════════════════════════════════════
```
