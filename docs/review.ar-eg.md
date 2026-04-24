# مراجعة تقنية: مشروع كشف الأخبار الكاذبة (Fake News Detection)

**المراجع:** Senior AI Engineer  
**التاريخ:** 2026-04-24  
**المستودع (Repository):** Fake-News-Detection  
**الفرع (Branch):** main

---

## الجزء 1 — مراجعة تقنية (بعيد عن المتطلبات)

### 1.1 هيكل المشروع

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

**التقييم:** الهيكل نضيف ومنطقي. فصل الـ notebooks عن الـ outputs وعن التطبيق خطوة كويسة. لكن مجلد `data/` مش موجود في الريبو — الداتا الخام والـ CSV المعالج مش معمولهم commit، وده معناه إن الـ notebooks مش هتتعمل re-run من clone جديد إلا لو حد حط الداتا يدويًا.

---

### 1.2 الداتا والاستكشاف (`01_exploration.ipynb`)

- **الداتا (Dataset):** Fake.csv + True.csv، واتدمجوا لحد 44,898 عينة.
- **توازن الكلاسات:** Fake 23,481 (52.3%) / Real 21,417 (47.7%) — قريب من المتوازن، وده كويس.
- **تحليل تكرار الكلمات:** اتعمل لكل كلاس لوحده. أكتر الكلمات كانت مسيطرة عليها كلمات إنجليزية عامة ومش مفيدة من غير إزالة stopwords (والـ stopwords اتتعامل معاها صح في الـ preprocessing).
- **المخرجات (Output):** ملف `dataset_stats.json` اتسجل بشكل مظبوط.

**مشاكل:**
- مفيش تحليل موثّق لـ null/missing values في الـ notebook.
- مفيش فحص للـ duplicate articles (داتا الأخبار كتير بيبقى فيها شبه-تكرار وده بيزوّد الدقة بشكل مضلل).
- مفيش تحليل لتوزيع أطوال النصوص (مفيد جدًا لاختيار max sequence length للـ LSTM).
- تحليل تكرار الكلمات اتعمل قبل التنضيف، فده بيقلل القيمة التحليلية.

---

### 1.3 المعالجة المسبقة (Preprocessing) (`02_preprocessing.ipynb`)

دالة `clean_text` بتعمل:

1. تحويل لحروف صغيرة (Lowercase)
2. إزالة آثار/علامات المصدر بالـ regex (`reuters|21st century wire`)
3. إزالة الأرقام
4. إزالة علامات الترقيم/الرموز الخاصة
5. Tokenization على whitespace
6. إزالة stopwords (NLTK English stopwords)
7. فلترة الكلمات القصيرة (length < 2)

**مشاكل:**
- **عدم اتساق قائمة الـ stopwords:** الـ notebook بيستخدم NLTK stopwords، بينما `streamlit_app.py` بيستخدم `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`. القايمتين مش زي بعض، وده معناه إن سلوك الـ preprocessing مختلف بين التدريب ووقت الـ inference. دي **مشكلة data leakage / train-serve skew** — ممكن التوقعات في التطبيق تتعمل على نص متعالج بطريقة مختلفة عن اللي الموديلات اتدربت عليها.
- **إزالة الأرقام مش لازمة وممكن تضر:** حذف كل الأرقام بيشيل أنماط رقمية ممكن تبقى مميزة (زي "2016 election" أو تواريخ معينة بتدي سياق).
- **مفيش lemmatization أو stemming.** مش شرط يبقى غلط، بس دي نقطة ممكن تتحسن.
- **دمج الخصائص (Feature combination):** دمج العنوان + النص في `content` استراتيجية معقولة، بس ممكن تضعف إشارات العنوان لوحده.

---

### 1.4 استخراج الخصائص (Feature Extraction)

**TF-IDF (مسار Logistic Regression):**
- `max_features=50,000`, `ngram_range=(1,2)` — إعدادات كويسة. الـ bigrams بتلقط تعبيرات من كلمتين.
- اتسجل في `vectorizer.pkl` واتحمّل صح في التطبيق.

**Word Embeddings (مسار LSTM):**
- Embeddings متعلَّمة (128-dim) متبنية من vocab التدريب.
- `vocab_size`: أعلى 10,000 كلمة تكرارًا، و `min_freq=2`.
- الـ sequences متقفولة على 200 token، ومعمولها zero-padding.
- **ملاحظة:** ده embedding متعلَّم (learned embedding)، مش pre-trained (Word2Vec, GloVe, BERT). المتطلب بيقول لازم embedding method — ده بيحقق روح المتطلب بس من غير تمثيلات جاهزة خارجية.

---

### 1.5 الموديل 1 — Logistic Regression

- **المدخلات (Input):** مصفوفة TF-IDF sparse (50K features، bigrams).
- **الهايبر باراميترز:** `max_iter=3000` — رقم كبير كفاية لضمان الـ convergence.
- **الأداء (Performance):**

| المقياس (Metric) | Fake  | Real  | Macro |
|------------------|-------|-------|-------|
| Precision        | 98.9% | 98.5% | 98.7% |
| Recall           | 98.7% | 98.7% | 98.7% |
| F1-score         | 98.8% | 98.6% | 98.7% |
| Accuracy         | **98.71%** | | |

- **قابلية التفسير (Explainability):** استخراج أعلى 5 معاملات (coefficients) من TF-IDF لكل prediction — منهجية سليمة.

**مشاكل:**
- مفيش tuning مذكور لقوة الـ regularization (`C`). الافتراضي `C=1.0` مستخدم من غير validation.
- مفيش cross-validation — تقسيمة واحدة 80/20 مش بتضمن الاستقرار، خصوصًا مع احتمال وجود near-duplicate articles في الداتا.
- دقة 98.71% على الداتا دي عالية بشكل مريب ومتوافق مع تقارير معروفة عن data leakage في Kaggle Fake/True news dataset (علامات مصدر زي "Reuters" لو لسه موجودة في النص ممكن تفرق Real بسهولة).

---

### 1.6 الموديل 2 — LSTM

**المعمارية (Architecture):**

```
Embedding(vocab_size, 128, padding_idx=0)
→ LSTM(128, 128, num_layers=2, dropout=0.3, batch_first=True)
→ Linear(128, 1)
→ BCEWithLogitsLoss
```

- **التدريب (Training):** 5 epochs، Adam (lr=0.001)، batch_size=32.
- **طول السلسلة (Sequence length):** 200 token (truncated/padded).
- **الأداء (Performance):**

| المقياس (Metric) | Fake  | Real  | Macro |
|------------------|-------|-------|-------|
| Precision        | 94.2% | 98.9% | 96.6% |
| Recall           | 99.1% | 93.2% | 96.2% |
| F1-score         | 96.6% | 96.0% | 96.3% |
| Accuracy         | **96.31%** | | |

**مشاكل:**
- **مفيش learning rate scheduling.** Adam بـ lr ثابت 0.001 لمدة 5 epochs ممكن ما يوصلش لأفضل convergence.
- **مفيش early stopping.** خطر الـ overfitting مش متراقَب، و5 epochs رقم اعتباطي.
- **مفيش منحنى validation loss متترسم.** فمش معروف الموديل overfit ولا لأ.
- **BiLSTM مش مستخدم.** BiLSTM غالبًا بيطلع أفضل من unidirectional في تصنيف النصوص من غير تكلفة كبيرة.
- **`dropout=0.3` بيشتغل بس بين طبقات الـ LSTM (num_layers=2)، مش قبل آخر Linear layer.** إضافة dropout قبل `fc` ممكن تفيد.
- الـ LSTM أقل من Logistic Regression بـ 2.4%، وده ملحوظ ومحتمل سببه مشكلة الـ data leakage (توكنز المصدر زي "Reuters" بتأثر قوي على TF-IDF واتشالت جزئيًا بس).

---

### 1.7 تطبيق Streamlit (`streamlit_app.py`)

**نِقَط كويسة:**
- UI نضيف ومقارنات جنب بعض بين الموديلات.
- عرض الثقة (Confidence) بـ Plotly bar charts.
- عرض أهمية الكلمات (TF-IDF coefficient × feature value).
- Error analysis عبر heuristic cross-validation.
- مؤشر Hard test mode.

**مشاكل:**

1. **اختلاف stopwords (bug خطير):** `clean_text` في `streamlit_app.py` بيستخدم `sklearn.ENGLISH_STOP_WORDS`، والـ notebooks بتستخدم NLTK stopwords. الموديلات اتدربت على NLTK stopwords، فـ inference بيستخدم فلتر vocab مختلف. ده بيقلل دقة التوقعات في الاستخدام الفعلي.

2. **تفسير احتمال LSTM معكوس في عرض الثقة:**
   ```python
   st.progress(float(lstm_prob))
   ```
   `lstm_prob` هو خرج sigmoid ناحية الكلاس 1 (Real). لو الموديل توقع Fake (prob < 0.5)، شريط الثقة هيظهر قيمة قليلة، مش الثقة الفعلية في توقع Fake. ده مضلل للمستخدم.

3. **`log_prob = max(log_model.predict_proba(vec)[0])`** — أخذ الـ max من احتمالات الكلاسين صح للعرض كـ "أعلى ثقة" بس المعنى مش دقيق. مثال: لو الموديل توقع Fake بثقة 60%، `max` هيرجع 60%، ولو توقع Real بـ 60% برضه هيرجع 60%. العرض كده بيضيع معلومة الثقة رايحة لأنهي كلاس.

4. **Hard test mode لغوي بحت وهش.** كلمة "government" لو ظهرت مرة واحدة بتفعّل "Hard / Realistic News Detected" بغض النظر عن السياق.

5. **كسر التعادل في heuristic model بيرجّح Real (label=1)** لما `fake_score == real_score`. ده بيعمل انحياز ضد اكتشاف الأخبار الكاذبة في الحالات الرمادية.

6. **مفيش تحقق من طول الإدخال.** التطبيق بيقبل نصوص قصيرة جدًا أو طويلة جدًا من غير تحذير.

7. **تحميل الموديلات على مستوى global من غير error handling.** لو ملف موديل ناقص، التطبيق هيقع عند التشغيل برسالة مش مفهومة.

8. **`streamlit` و `plotly` ناقصين من `requirements.txt`.** التطبيق مش هيقدر يتسطّب من ملف المتطلبات لوحده.

---

### 1.8 ملف المتطلبات (Requirements File)

```
pandas
numpy
scikit-learn
nltk
torch
```

**Dependencies ناقصة:**
- `streamlit` — مطلوب لـ `streamlit_app.py`
- `plotly` — مطلوب لـ `streamlit_app.py`
- `joblib` — متستورد بشكل صريح (مع إنه بييجي ضمن scikit-learn، بس كتابته صراحة Best Practice)

مفيش version pins متحطوطة. ده مقبول لمشروع طلابي، بس خطر في production — إصدارات `torch` الكبيرة ممكن تكسر توافق الـ API.

---

### 1.9 ملخص جودة الكود

| الجزء (Area) | التقييم (Assessment) |
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

## الجزء 2 — التحقق من المتطلبات (Requirements Validation)

### المتطلبات العامة الإلزامية

#### REQ-G1: تطبيق تقنيات المعالجة المسبقة للنص

**الحالة: PASS**

متنفّذ في `02_preprocessing.ipynb`:
- تحويل لحروف صغيرة ✓
- إزالة علامات الترقيم ✓
- إزالة stopwords (NLTK) ✓
- Tokenization (whitespace split) ✓

فجوة بسيطة: مفيش lemmatization، بس ده مش مطلوب.

---

#### REQ-G2: استخدام طريقتين لاستخراج الخصائص — TF-IDF (إجباري) + طريقة embedding واحدة

**الحالة: PARTIAL PASS**

- TF-IDF: متنفّذ بالكامل بـ 50K features و bigrams ✓
- Embedding method: Embeddings متعلَّمة داخل LSTM (128-dim) ✓

**فجوة:** المتطلب كاتب "Word2Vec, GloVe, Transformer embeddings" كأمثلة لطرق embedding. التنفيذ الحالي بيستخدم embeddings معمولة random initialization واتدربت end-to-end مع الـ LSTM بدل ما تكون embedding method مستقلة قابلة لإعادة الاستخدام. الـ embeddings ضمنيًا جزء من الـ LSTM مش مرحلة Feature Extraction منفصلة. ده معماريًا صحيح، بس مش بيظهر استخدام Word2Vec/GloVe/BERT بشكل صريح كـ feature extraction step.

---

#### REQ-G3: تنفيذ على الأقل موديلين — واحد baseline + واحد advanced

**الحالة: PASS**

- Baseline model: Logistic Regression ✓
- Advanced model: LSTM (PyTorch, 2-layer, 128-hidden) ✓

---

#### REQ-G4: عمل مقارنة واضحة بين الموديلات

**الحالة: PASS**

المقارنة موثّقة في `03_modeling.ipynb` وكمان في تطبيق Streamlit:
- مقارنة الدقة: LR 98.71% vs LSTM 96.31% ✓
- سرد نقاط القوة/الضعف: LR أبسط وأثبت؛ LSTM عنده fake recall أحسن (99.07%) بس real recall أقل ✓

---

#### REQ-G5: توفير مقاييس التقييم — Accuracy, Precision/Recall/F1, Confusion Matrix

**الحالة: PARTIAL PASS**

- Accuracy: موجودة للموديلين ✓
- Precision / Recall / F1-score: تقارير تصنيف كاملة محفوظة كـ JSON ✓
- Confusion Matrix: **مش موجودة في أي output للـ notebooks ولا artifact متسجل.** والمتطلبات بتطلبها صراحة. مفيش plot أو جدول.

**فجوة:** مفيش confusion matrix visualization أو حتى matrix مطبوعة في الـ notebooks، ولا التطبيق، ولا ملفات النتائج.

---

#### REQ-G6: تسليم تقرير نهائي

**الحالة: PASS**

`report/Fake_News_Detection_Report.pdf` (4.1 MB) موجود ✓

---

### متطلبات خاصة بالمشروع

#### REQ-P1: بناء نظام يصنّف المقالات Fake أو Real

**الحالة: PASS**

متنفّذ end-to-end: بايبلاين تدريب في الـ notebooks، و inference في تطبيق Streamlit. LR و LSTM شغالين تصنيف ثنائي (0=Fake, 1=Real) ✓

---

#### REQ-P2: فهم الأنماط بين fake و real

**الحالة: PASS**

- تحليل معاملات TF-IDF بيوضح الـ n-grams المميزة لكل كلاس ✓
- Heuristic pattern detection في التطبيق بيميز لغة مبالغ فيها (breaking, shocking, unbelievable) ضد لغة رسمية/واقعية (according to, announced, reported) ✓
- إحصائيات الداتا محفوظة (توزيع fake/real) ✓

---

#### REQ-P3: شرح ليه الخبر اتصنّف fake أو real

**الحالة: PASS**

دالة `explain()` في التطبيق بتطلع لحد 8 كلمات مع signed TF-IDF impact scores:
- قيمة سالبة → إشارة Fake (red)
- قيمة موجبة → إشارة Real (green)

وده مطابق لصيغة المتطلب: `Fake Reason: "you won't believe", "breaking"` ✓

---

#### REQ-P4: استخراج كلمات مهمة (زي shocking, breaking)

**الحالة: PASS**

متنفّذ عن طريق:
1. استخراج معاملات TF-IDF في `explain()` ✓
2. قائمة كلمات heuristic (fake_signals, real_signals) في `heuristic_label()` ✓

---

#### REQ-P5: تحديد أنماط مشتركة في الأخبار الكاذبة

**الحالة: PASS**

متغطّي عن طريق:
- أعلى features لكل كلاس في TF-IDF (تحليل notebook) ✓
- Heuristics: fake news فيها clickbait/لغة عاطفية؛ real news فيها لغة رسمية/واقعية ✓

---

#### REQ-P6: إحصائيات بسيطة (زي % fake vs real)

**الحالة: PASS**

`dataset_stats.json`:
- Fake: 23,481 (52.30%)
- Real: 21,417 (47.70%)

ومتعرضة في `01_exploration.ipynb` ✓

---

#### REQ-P-TASK1: Preprocessing — lowercase, punctuation removal, stopwords, tokenization

**الحالة: PASS** (مع ملاحظة عدم اتساق stopwords فوق)

---

#### REQ-P-TASK2: Feature Extraction — TF-IDF + Word embeddings

**الحالة: PARTIAL PASS** (راجع REQ-G2 فوق)

---

#### REQ-P-TASK3: Baseline Model — Logistic Regression

**الحالة: PASS** ✓

---

#### REQ-P-TASK4: Advanced Model — LSTM أو Transformer

**الحالة: PASS** (LSTM متنفّذ) ✓

---

#### REQ-P-TASK5: Evaluation — Accuracy, Confusion Matrix, Precision/Recall/F1, مقارنة الموديلات

**الحالة: PARTIAL PASS**

- Accuracy ✓
- Precision/Recall/F1 ✓
- Model comparison ✓
- **Confusion Matrix: MISSING** ✗

---

#### REQ-P-TASK6: Report — وصف المشكلة، الداتا، preprocessing، الموديلات، النتائج، الخلاصة

**الحالة: PASS**

PDF موجود في `report/Fake_News_Detection_Report.pdf` ✓ (المحتوى نفسه مش متراجع بشكل مستقل غير وجود الملف)

---

## الجزء 3 — الخلاصة

### الالتزام بالمتطلبات (Requirements Compliance)

| المتطلب (Requirement) | الحالة (Status) | ملاحظات (Notes) |
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

**الإجمالي: 14/15 متطلب متحقق. 1 FAIL صريحة (Confusion Matrix). 2 PARTIAL PASS (نوع الـ embedding، وموضوع الـ confusion matrix كمان داخل في مهمة التقييم).**

---

### أهم مشاكل لازم تتصلّح

1. **[CRITICAL] Confusion Matrix مش موجودة** — دي مطلوبة صراحة. ضيف `sklearn.metrics.confusion_matrix` وارسمها للموديلين في `03_modeling.ipynb`.

2. **[CRITICAL] اختلاف stopwords library** — `streamlit_app.py` بيستخدم sklearn stopwords والـ notebooks بتستخدم NLTK. لازم توحّد نفس الليست في الاتنين. أسهل حل: خَلّي التطبيق يستخدم NLTK stopwords زي التدريب.

3. **[IMPORTANT] `streamlit` و `plotly` ناقصين في `requirements.txt`** — ضيف الاتنين.

4. **[IMPORTANT] عرض ثقة LSTM مضلل لما يتوقع Fake** — اعرض `1 - lstm_prob` لما التوقع يكون Fake.

5. **[MINOR] احتمال data leakage** — آثار المصدر (reuters) اتشالت جزئيًا. دقة 98.71% ممكن تبقى inflated. فكّر تعمل re-run بعد تنضيف أقوى لآثار المصدر علشان تتأكد من الأداء الحقيقي.

6. **[MINOR] مفيش cross-validation** — k-fold split هيدي تقدير أداء أكثر ثباتًا.

7. **[MINOR] الداتا مش معمولها commit** — ضيف سكريبت تحميل أو Kaggle API call علشان المشروع يبقى reproducible.
