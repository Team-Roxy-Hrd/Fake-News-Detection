# Natural Language Processing

## Project Ideas

### General Objective

Each team will build a complete NLP pipeline that includes:

- Text preprocessing
- Feature extraction
- Baseline modeling
- Advanced modeling
- Evaluation and comparison
- (Optional with bonus) enhancement or extension

### General Requirements (MANDATORY for ALL projects)

Each team must:

1. Apply text preprocessing techniques.
2. Use two feature extraction methods:
	- TF-IDF (required)
	- One embedding method (Word2Vec, GloVe, Transformer embeddings, ....)
3. Implement at least TWO models:
	- One baseline model (e.g., Logistic Regression, Naive Bayes)
	- One advanced model (e.g., LSTM or Transformer)
4. Perform a clear comparison between models:
	- Accuracy (or relevant metric)
	- Strengths & weaknesses
5. Provide evaluation metrics:
	- Accuracy       - Precision / Recall / F1-score        - Confusion Matrix
6. Submit a final report.

## Project 3: Fake News Detection

### 1. Objective

- Build a system that classifies news articles as Fake or Real.
- Understand patterns in fake vs real news.
- Explain why a news article is classified as fake or real.

“Explaining predictions” means:

Finding reasons behind the model decision such as important words or patterns that led to classification.

Example of Classification:

- News:      "Breaking: You won’t believe what happened next!"
- Basic output (✘):
  - Fake
- Improved output (✓) → you must do like this:
  - Fake
  - Reason: "you won’t believe", "breaking"

You must extract:

- Important keywords (e.g., shocking, breaking)
- Common patterns across fake news
- Simple statistics (e.g., % of fake vs real news)

Common patterns mean:

The system should identify patterns such as:

- Fake news often uses exaggerated words (e.g., "shocking", "unbelievable")
- Real news uses more formal and factual language

### 2. Dataset: From your choice.

Examples:

- Fake News datasets (Kaggle)

### 3. Tasks

1. Preprocessing
	- Convert text to lowercase
	- Remove punctuation
	- Remove stopwords
	- Tokenization
2. Feature Extraction
	- TF-IDF
	- Word embeddings (Word2Vec / GloVe / BERT embeddings)
3. Baseline Model
	- Logistic Regression
4. Advanced Model
	- LSTM or Transformer (e.g., BERT)
5. Evaluation
	- Accuracy  - Confusion Matrix -  Precision, Recall, F1-score
	- Compare both models
6. Report (write what you did in detail)

Include:

- Problem description & Dataset description
- Preprocessing steps
- Models used
- Results and comparison
- Conclusion