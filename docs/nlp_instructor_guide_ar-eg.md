# اكتشاف الأخبار الكاذبة — تعمّق NLP (Fake News Detection — NLP Deep Dive)
### دليل مُدرّس محترف لكل فكرة، قرار، وبديل (A Professional Instructor's Guide to Every Concept, Decision, and Alternative)

---

## فهرس المحتويات (Table of Contents)

1. [إحنا بنحل مشكلة إيه بالظبط؟ (What Problem Are We Actually Solving?)](#1-what-problem-are-we-actually-solving)
2. [الداتا سِت — هي إيه وليه مهمة (The Dataset — What It Is and Why It Matters)](#2-the-dataset)
3. [استكشاف وتحليل مبدئي للداتا — ليه لازم تبص قبل ما تعمل موديل (Exploratory Data Analysis — Why You Must Look Before You Model)](#3-exploratory-data-analysis)
4. [تنضيف وتجهيز النص — الأساس اللي كل حاجة واقفة عليه (Text Preprocessing — The Foundation Everything Stands On)](#4-text-preprocessing)
5. [استخراج الخصايص/الـ Features — إزاي نعلّم الماكينة تقرا (Feature Extraction — Teaching Machines to Read)](#5-feature-extraction-teaching-machines-to-read)
6. [الموديل 1 — Logistic Regression مع TF-IDF (Model 1 — Logistic Regression with TF-IDF)](#6-model-1--logistic-regression-with-tf-idf)
7. [الموديل 2 — LSTM مع تمثيلات GloVe (Model 2 — LSTM with GloVe Embeddings)](#7-model-2--lstm-with-glove-embeddings)
8. [التقييم — بنقيس إيه اللي مهم فعلاً؟ (Evaluation — Measuring What Actually Matters)](#8-evaluation--measuring-what-actually-matters)
9. [قابلية التفسير — الموديل قال كده ليه؟ (Explainability — Why Did the Model Say That?)](#9-explainability--why-did-the-model-say-that)
10. [مقارنة الموديلات والدروس المستفادة (Model Comparison and Lessons Learned)](#10-model-comparison-and-lessons-learned)
11. [واجهة الويب — نجمع كل حاجة مع بعض (The Web UI — Bringing It All Together)](#11-the-web-ui)
12. [المحترفين يعملوا إيه بعد كده؟ (What Would Professionals Do Next?)](#12-what-would-professionals-do-next)

---

## 1. What Problem Are We Actually Solving?

### التاسك (The Task)

**Fake news detection** هو مشكلة **تصنيف نصوص ثنائي (binary text classification)**. يعني قدّامنا مقال أخبار (نص)، والموديل لازم يتوقّع هل هو `REAL` (0) ولا `FAKE` (1).

الموضوع شكله بسيط، بس هو في الحقيقة مُخادِع شوية. الصعوبة مش إننا نفرّق بين كلام “هُري” وكلام صح — الاتنين غالبًا مكتوبين إنجليزي كويس وبشكل سليم نحويًا. الصعوبة إننا نلقط **إشارات لغوية وأسلوبية** دقيقة بتفرّق بين صحافة موثوقة وبين معلومات مضللة.

### ليه المشكلة دي صعبة من منظور NLP؟ (Why This Problem Is Hard in NLP Terms)

| التحدّي (Challenge) | ليه مهم؟ (Why It Matters) |
|-----------|----------------|
| الاتنين لغتهم سليمة وسلسة | قواعد بسيطة/حوارات نحوية مش هتنفع |
| التناول السياسي ممكن يبقى “وجهة نظر” | نفس الحقيقة ممكن تتصنّف Real أو Fake حسب طريقة الصياغة |
| تشابه في الكلمات كبير | كلمات زي "Trump" و"election" و"government" موجودة في الاتنين |
| سخرية/إيروني | الموديلات العميقة بتتلخبط مع الأساليب البلاغية |
| تغيّر الدومين (Domain shift) | موديل متدرّب على أخبار 2016 ممكن يفشل مع أخبار 2024 |
| تحيّز المصدر في الداتا | بعض الداتا بتلصّق اللابل بالمصدر (Breitbart = fake) مش بالمحتوى |

### إيه الإشارات اللي بتبان في الأخبار الفيك؟ (What Signals Does Fake News Carry?)

الأبحاث + الـ EDA على الداتا دي بيأكدوا شوية إشارات كويسة:

1. **Lexical markers** — كلمات زي "shocking"، "breaking"، "you won't believe" وكلام نظريات مؤامرة
2. **Reporting style** — الأخبار الـ Real كتير بتستخدم نسب الكلام لمصدر ("said"، "according to"، "reported")
3. **Vocabulary diversity** — المقالات الفيك ساعات بتكرر نفس الكلمات العاطفية/المستفزّة
4. **Article length** — الأخبار الـ Real بتبقى أطول شوية (تفاصيل وسرد أكتر)
5. **Sensational phrasing** — علامات تعجب كتير، حروف كابيتال، صياغات Clickbait

الإشارات دي معناها إن المشكلة **ممكن تتحلّ كويس جدًا بخصايص لغوية/معجمية (lexical features) بس** — وده سبب إن حتى Logistic Regression بسيط يوصل ~94% Accuracy هنا.

---

## 2. The Dataset

### إيه اللي عندنا؟ (What We Have)

| الملف (File) | الحجم (Size) | عدد المقالات (Articles) |
|------|------|----------|
| `Fake.csv` | 59.9 MB | 23,490 مقال فيك |
| `True.csv` | 51.1 MB | 21,418 مقال حقيقي |
| **Merged** | — | **44,908 إجمالي** |
| بعد إزالة التكرار (deduplication) | — | **44,049** |

**الخصايص الخام (Raw features):**
- `title` — عنوان الخبر
- `text` — نص الخبر
- `subject` — تصنيف/قسم الخبر
- `date` — تاريخ النشر

بعد الـ preprocessing بنضيف عمود `label`: `0 = Real`, `1 = Fake`.

### ليه توازن الكلاسات مهم هنا؟ (Why Class Balance Matters Here)

الداتا تقريبًا **52% Real / 48% Fake** — وده متوازن جدًا مقارنة بالواقع. في الحقيقة غالبًا هتلاقي أخبار حقيقية أكتر بكتير من الأخبار الفيك على أي منصة.

**ليه بنستخدم `class_weight="balanced"` برضه؟** حتى فرق 52/48 ممكن يخلّي الكلاسيـفاير يميل سنة ناحية الأغلبية. الـ class weights بتخلي عقوبة الغلط في الكلاسين شبه بعض. وده مهم لأن **تكلفة إنك تتهم خبر حقيقي إنه فيك** (رقابة/ظلم) ممكن تكون مؤذية زي **تكلفة إنك تسيب خبر فيك يعدّي** (نشر تضليل).

### قرار دمج العنوان مع النص (The Combined Text Design Decision)

في Notebook الـ preprocessing بنجمع `title` و`text` في حقل واحد اسمه `combined_text` قبل التنضيف.

**ليه؟** لأن العنوان لوحده بيشيل إشارات قوية جدًا:
- عناوين Real: هادية وموضوعية ("Senate Passes Infrastructure Bill")
- عناوين Fake: مثيرة/مبالغ فيها ("SHOCKING: Government HIDES Truth About Vaccines")

لو شيلنا العنوان هنضيّع سيجنال قوي. لما نجمع الاتنين، الموديل بيشوف الصورة كاملة.

**بدائل ممكنة (Alternative approaches):**
- **Dual-input model** — مدخل للعناوين ومدخل للمتن وبعدين ندمجهم Late fusion. أعقد، وتحسُّن بسيط.
- **Title-only model** — سريع وخفيف للنشر. بس هيخسر سيجنال الجسم.
- **Weighted concatenation** — تكرّر العنوان N مرات قبل الدمج علشان تزود وزنه. هاك بسيط وممكن ينفع أحيانًا.

---

## 3. Exploratory Data Analysis

### ليه الـ EDA مش رفاهية؟ (Why EDA Is Not Optional)

قبل ما تبني أي موديل NLP لازم تفهم الداتا بتاعتك. الـ EDA بيجاوب على أسئلة زي:
1. الداتا متوازنة ولا لأ؟ → يحدد هل نستخدم `class_weight`
2. إيه الكلمات/الإشارات المميزة لكل كلاس؟ → يساعد في feature engineering
3. هل في مشاكل جودة داتا؟ → يحدد احتياج الـ preprocessing
4. توزيع الأطوال عامل إزاي؟ → يحدد قرارات truncation/padding في الموديلات التسلسلية

### أهم نتايج الـ EDA في المشروع ده (Key EDA Findings in This Project)

**تحليل تكرار الكلمات (بعد إزالة stopwords):**

| أعلى كلمات في Fake | أعلى كلمات في Real |
|---------------------|---------------------|
| trump (89k) | said (183k) |
| people (62k) | trump (106k) |
| obama (45k) | government (91k) |
| media (41k) | new (78k) |
| american (38k) | us (72k) |

كلمة "said" من أقوى الإشارات اللي بتميّز **الأخبار الحقيقية**. الصحفيين بيستخدموا إسناد لمصدر كتير ("he said"، "officials said"). المقالات الفيك كتير بتقدّم الكلام كأنه حقيقة مطلقة من غير ما تقول “مين قال”.

**توزيع الأطوال (Length distribution):**
- Fake news: حوالي ~511 كلمة متوسط (raw)، ~241 بعد التنضيف
- Real news: حوالي ~581 كلمة متوسط (raw)، ~291 بعد التنضيف
- الـ Real غالبًا أطول سنة (تفاصيل ومصادر أكتر)

**تنوع المفردات (Vocabulary diversity / unique word ratio):**
- Real: تنوّع أعلى شوية
- Fake: تكرار أكتر لكلمات معينة (خصوصًا العاطفية)

### الـ EDA بيقولنا إيه عن المودلينج؟ (What EDA Tells Us About Modeling)

النتايج دي بتأكد إن **الـ lexical features هي المسيطرة**. يعني “إيه الكلمات المستخدمة” أهم من “ترتيب الكلمات”. وده سبب إن Logistic Regression + TF-IDF بيطلع قريب جدًا من LSTM هنا: الموديل مش محتاج يفهم sequence بعمق علشان يميز الفيك في الداتا دي.

---

## 4. Text Preprocessing

### بايبلاين التنضيف بالكامل (The Full Pipeline)

```python
def clean_text(text):
	text = text.lower()                          # Step 1: Lowercase
	text = re.sub(r'http\S+|www\S+', '', text)   # Step 2: Remove URLs
	text = re.sub(r'\S+@\S+', '', text)          # Step 3: Remove emails
	text = re.sub(r'[^a-z\s]', ' ', text)        # Step 4: Remove punctuation/numbers
	text = re.sub(r'\s+', ' ', text).strip()     # Step 5: Normalize whitespace
	tokens = word_tokenize(text)                 # Step 6: Tokenize
	tokens = [t for t in tokens                 # Step 7: Remove stopwords
			  if t not in stopwords.words('english')]
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Step 8: Lemmatize
	tokens = [t for t in tokens if len(t) > 2]  # Step 9: Filter short tokens
	return ' '.join(tokens)
```

تعالى نفصّص كل خطوة.

---

### Step 1: Lowercasing

**إيه اللي بيحصل؟** بنحوّل النص كله لحروف صغيرة.

**ليه؟** موديلات NLP بتعتبر "Trump" و"TRUMP" و"trump" 3 توكنز مختلفين لو ماوحّدناهمش. ده بيكبّر الـ vocabulary من غير قيمة.

**بديل — نخلي الـ case:** لو الكابيتال ليه معنى (زي BREAKING NEWS كـ emphasis، أو اختصارات زي FBI). في التاسك دي، lowercasing غالبًا مناسب لأننا مهتمين بهوية الكلمة مش شكلها.

**إمتى ماينفعش lowercase؟** في مهام زي Named Entity Recognition (NER)، لأن "Apple" الشركة غير "apple" الفاكهة.

---

### Step 2 & 3: إزالة الروابط والإيميلات (URL and Email Removal)

**إيه؟** بنشيل أنماط زي `http://...` و`www...` و`user@domain.com`.

**ليه؟** غالبًا روابط/إيميلات ضوضاء في تصنيف المحتوى. والأهم: دومينات معينة (breitbart.com, reuters.com) ممكن تخلي الموديل “يغش” ويحفظ سمعة المصدر بدل مايتعلم إشارات المحتوى. إحنا عايزينه يتعلم من **المحتوى** مش من **هوية المصدر**.

**بديل — نطلع الدومين كـ feature:** قوي جدًا بس بيزود تحيّز المصدر، وبيخلي الموديل هش مع مصادر جديدة.

---

### Step 4: إزالة علامات الترقيم والأرقام (Punctuation and Number Removal)

**إيه؟** بنستبدل أي حاجة غير حرف/مسافة بمسافة.

**ليه؟** علامات زي "!" أو "???" ممكن تبقى إشارة sensationalism، بس بتبقى sparse وصعب على موديلات بسيطة تستغلها. والأرقام زي (2016, 2024) بتفتت الـ vocabulary من غير فايدة كبيرة هنا.

**بديل — نخلي الترقيم كـ features:** مفيد في sentiment/sarcasm. ممكن تحسب عدد `!` و`?` ونسبة ALL_CAPS.

**بديل — نخلي الأرقام:** لو الزمن/تواريخ معينة مهمة في المشكلة. في التصنيف العام هنا، إزالتها بتبسّط الدنيا.

---

### Step 6: Tokenization

**إيه؟** بنقسم النص لتوكنز (كلمات).

**ليه؟** باقي الخطوات (stopwords، lemmatization، vectorization) بتشتغل على توكنز.

**NLTK `word_tokenize` vs بدائل:**

| الطريقة | بتعمل إيه | تستخدمها إمتى |
|--------|-----------|----------------|
| `str.split()` | تقسيم على المسافات | بروتوتايب سريع بس |
| `word_tokenize` (NLTK) | قواعد بتتعامل مع contractions | NLP إنجليزي عام |
| spaCy tokenizer | أقوى وبيغطي edge cases | شغل production |
| Subword tokenization | بيكسر كلمات نادرة لأجزاء | موديلات Transformers (BERT, GPT) |

في المشروع ده، `word_tokenize` مناسب لأنه بيتعامل كويس مع كلمات زي "won't".

---

### Step 7: Stopword Removal

**إيه؟** بنشيل كلمات وظيفية شائعة (the, is, and, of, to, ...).

**ليه؟** غالبًا مالهاش قيمة تمييزية، وبتنفّخ الـ vocabulary وبتطوّل الـ sequences.

**تأثيرها على TF-IDF:** الـ IDF أصلاً بيدّي وزن شبه صفر للـ stopwords لأنها بتظهر في معظم المستندات. فالإزالة هنا شبه redundant، بس بتقلل الحجم وتسرّع.

**تأثيرها على LSTM:** من غير إزالة، كتير من الـ 300 توكن يبقوا "the" و"of"… فالموديل بيضيّع قدرة كبيرة وهو بيتعلم يتجاهلهم.

**بديل — ما تشيلش stopwords مع deep learning:** مع Transformers زي BERT/RoBERTa، إزالة stopwords ممكن **تبوّظ** الأداء لأنهم بيفهموا السياق والتركيب. مثال: "He did NOT say that" vs "He did say that" — لو شيلت "NOT" انت كسرت المعنى.

**الخلاصة المهمة:** إزالة stopwords heuristic مناسب لـ **bag-of-words**، وممكن يضر موديلات بتفهم الـ sequence.

---

### Step 8: Lemmatization

**إيه؟** بنرجّع الكلمة لجذرها القاموسي (lemma):
- "running" → "run"
- "studies" → "study"
- "wolves" → "wolf"

**ليه؟** تقليل الـ vocabulary. بدل ما "run/runs/ran/running" يبقوا 4 features، يبقوا feature واحد قوي.

**Lemmatization vs Stemming:**

| | Lemmatization | Stemming (Porter/Snowball) |
|--|---------------|---------------------------|
| الطريقة | قاموس/تحليل صرفي | قص لواحق بقواعد |
| الناتج | كلمات مفهومة | ساعات بيطلع شِبه كلمة |
| الدقة | أعلى (والأفضل مع POS tags) | أقل |
| السرعة | أبطأ | سريع جدًا |
| مناسب لـ | قابلية قراءة/شرح | سرعة على نطاق كبير |

**ليه lemmatization هنا؟** لأن الـ UI عنده جزء شرح كلمات، فعايزين مخرجات مفهومة للمستخدم.

**ملحوظة عن `WordNetLemmatizer`:** من غير POS tags بيعتبر الكلمة noun افتراضيًا، فممكن يفوّت حالات.

```python
lemmatizer.lemmatize("running", pos='v')  # → "run"
lemmatizer.lemmatize("running")           # → "running" (غلط لأنه اتعامل كاسم)
```

---

### Step 9: فلترة التوكنز القصيرة (Short Token Filtering)

**إيه؟** بنشيل التوكنز اللي طولها ≤ 2.

**ليه؟** غالبًا بتطلع artifacts (زي "s" من الملكية، أو أجزاء من contractions)، وبتلوّث الـ vocabulary.

---

### القاعدة الذهبية: اعمل Fit على الداتا بتاعة التدريب بس (Fit on Training Data Only)

بعد التنضيف، عندنا خطوة مهمة في TF-IDF وTokenizer بتوع Keras: **تثبيت الـ vocabulary (fitting)**.

**القاعدة:** اعمل fit على الـ training فقط. وبعدين transform للـ train والـ test بنفس الـ vocabulary.

```python
# CORRECT:
vectorizer.fit(X_train)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)   # Uses training vocabulary

# WRONG (data leakage):
vectorizer.fit(X_train + X_test)            # Test vocabulary bleeds in
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

**ليه ده مهم؟** لو شميت الـ test في الـ fit، الـ IDF هيتحسب باستخدام تكرارات من الـ test، وده اسمه **data leakage**: بيزوّد الدقة بشكل وهمي ومش هيحصل في الإنتاج.

---

## 5. Feature Extraction — Teaching Machines to Read

### التحدّي الأساسي (The Core Challenge)

الموديلات بتتعامل مع أرقام. النص رموز. **Feature extraction** هو الكوبري بين اللغة والرياضة.

في 3 فلسفات رئيسية للتمثيل:

| Approach | Representation | Captures |
|----------|---------------|----------|
| Bag of Words / TF-IDF | فيكتور sparse من أوزان/عدّ الكلمات | وجود الكلمات وتكرارها |
| Word Embeddings (Word2Vec, GloVe) | فيكتور dense لكل كلمة | تشابه معنوي |
| Contextual Embeddings (BERT) | فيكتور لكل كلمة **في سياقها** | معنى سياقي كامل |

المشروع ده بيستخدم (1) و(2).

---

### TF-IDF: الماث اللي وراه (The Mathematics Behind It)

**TF-IDF** (Term Frequency–Inverse Document Frequency) بيقيس “الكلمة دي مهمة قد إيه في الدوكيومنت دي” بالنسبة لمجموعة الدوكيومنتس.

**Term Frequency (TF):**
```
TF(t, d) = count of term t in document d / total terms in document d
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
```
- `N` = عدد الدوكيومنتس
- `df(t)` = كام دوكيومنت فيها الكلمة `t`

الكلمة الشائعة جدًا (زي "the") → `df ≈ N` → `log(1)=0` → IDF صغير.

**TF-IDF Score:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**المعنى:** كلمة بتتكرر جوه مقال معيّن بس نادرة في باقي الكوربس → TF-IDF عالي → بتبقى “مميّزة”.

**مثال في الفيك نيوز:**
- "said" شائعة في real → IDF قليل
- "illuminati" نادرة عمومًا بس بتظهر في مقالات مؤامرة → TF-IDF عالي في النوع ده

---

### ليه 5,000 Feature؟ (Why 5,000 Features?)

`max_features=5000` بيخلّينا نحتفظ بأكتر 5000 term شيوعًا (حسب document frequency).

**الموازنة:**
- **قليل قوي** → هتفوّت كلمات نادرة بس قوية ("chemtrails", "deepstate")
- **كتير قوي** → أبعاد ضخمة، training أبطأ، وضوضاء من كلمات نادرة جدًا/أخطاء

5000 هنا حل عملي كويس، وبعد رقم معين العائد بيقل.

---

### ليه Bigrams؟ `ngram_range=(1,2)`

Unigrams: كلمة واحدة ("breaking")
Bigrams: كلمتين ("breaking news")

في الفيك نيوز، bigrams بتلقط عبارات أقوى من كلمة منفردة:

| Unigram | Bigram (أوضح) |
|---------|----------------|
| "breaking" | "breaking news" |
| "deep" | "deep state" |

**ليه مش trigrams؟** بتكبر الـ vocabulary جدًا، وبتبقى sparse زيادة، والفائدة غالبًا أقل من تكلفتها في التصنيف ده.

---

### GloVe Embeddings: نعلّم الماكينة تشابه المعاني (Teaching Semantic Similarity)

**GloVe** بيطلع vectors dense للكلمات مبنية على co-occurrence statistics. كل كلمة ليها فيكتور 100 رقم (هنا).

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
cosine_similarity(vector("fake"), vector("false")) ≈ 0.82
```

**فكرة التدريب:** بيقلّل فرق بين dot-product وبين `log(co-occurrence)` عبر الكوربس.

**ليه pretrained؟** تدريب embeddings من الصفر محتاج مليارات توكنز. الداتا هنا (~11 مليون كلمة تقريبًا) مش كفاية. GloVe 6B متدرّب على 6 مليار توكنز وبيجيب معرفة لغوية “جاهزة”.

**ليه `trainable=True`؟** علشان يعمل fine-tuning للدومين بتاع الأخبار. بس في مخاطرة **catastrophic forgetting** لو اتدرّب زيادة، وEarly stopping بيقلّلها.

**بديل: `trainable=False`** أسرع ويحافظ على المعاني العامة، ومناسب أكتر لما الداتا قليلة جدًا، بس أضعف في domain adaptation.

---

## 6. Model 1 — Logistic Regression with TF-IDF

### Logistic Regression يعني إيه؟ (What Is Logistic Regression?)

رغم الاسم، ده موديل **تصنيف** مش Regression. بيتعلّم حد فاصل خطي (linear decision boundary) وبعدين يحوّل السكور لاحتمال بالـ sigmoid.

**المعادلة:**

لو عندك فيكتور TF-IDF اسمه `x` أبعاده 5000:
```
z = w₀ + w₁x₁ + w₂x₂ + ... + w₅₀₀₀x₅₀₀₀
P(fake) = sigmoid(z) = 1 / (1 + e^(-z))
```

كل وزن `wᵢ` بيمثل تأثير feature `i`:
- `wᵢ > 0` → يميل ناحية "FAKE"
- `wᵢ < 0` → يميل ناحية "REAL"

**أمثلة من اللي الموديل غالبًا بيتعلمه:**
- `w("said") < 0` → attribution → إشارة Real
- `w("shocking") > 0` → sensationalism → إشارة Fake
- `w("reported") < 0` → لغة صحافة → Real
- `w("conspiracy") > 0` → مواضيع مؤامرة → Fake

### ليه Logistic Regression ممتاز للنصوص؟ (Why Logistic Regression for Text?)

| Property | ليه مهم لـ NLP |
|----------|----------------|
| Linear in feature space | الـ n-grams نفسها بتدي “تعقيد” كفاية |
| Probabilistic output | بيطلع احتمال مش بس 0/1 |
| Interpretable coefficients | تقدر تشرح “ليه” بسهولة |
| بيستحمل أبعاد كبيرة | شغال كويس مع sparse vectors |
| Regularization | L2 بيقلّل overfitting |
| سريع | تدريب/توقع سريع جدًا |

### ليه مش موديلات تانية؟ (Why Not Other Classifiers?)

**Naive Bayes:** سريع ومش وحش، بس بيفترض independence بين الكلمات (وده مش صحيح).

**SVM:** قوي جدًا في النصوص، وغالبًا منافس لـ LR. الاختيار بينهم غالبًا تجريبي.

**Random Forest / Gradient Boosting:** شغالين بس أبطأ وأقل ملاءمة للـ sparse high-dim النصي.

**الخلاصة:** في bag-of-words text classification، Logistic Regression واحد من أقوى الـ baselines.

---

### Cross-Validation: ليه 5-Fold؟ (Why 5-Fold?)

الـ notebook بيعمل train/test split وكمان 5-fold CV.

**ليه CV؟** split واحد ممكن يبقى “محظوظ” أو “وحش”. الـ CV بيقسّم الداتا 5 مرات ويجيب متوسط الأداء.

```
F1 macro = 0.939 ± 0.003
```

الـ ± 0.003 (standard deviation) معناها إن الأداء **مستقر** ومش بيتقلب جامد حسب العيّنة.

**ليه 5 بالذات؟** توازن كويس بين الدقة والتكلفة الحسابية (industry standard). 10-fold أدق بس أغلى.

---

## 7. Model 2 — LSTM with GloVe Embeddings

### يعني إيه LSTM؟ (What Is an LSTM?)

**LSTM (Long Short-Term Memory)** هو نوع من RNN معمول علشان يتعامل مع **Sequences** وهو بيعرف “يفتكر” و“ينسى” بشكل متحكَّم فيه لمسافات طويلة.

**مشكلة الـ RNN العادي:** بيحصل **vanishing gradient** عبر timesteps كتير، فالموديل بينسى بدايات النص مع الطول.

### LSTM بيحل ده إزاي؟ (How LSTM Solves This)

بيستخدم 3 gates (طبقات بتطلع قيم بين 0 و1) للتحكم في تدفق المعلومات:

1. **Forget gate** — يقرر يمسح إيه من الـ cell state
	```
	f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
	```

2. **Input gate** — يقرر يضيف إيه جديد
	```
	i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
	c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
	```

3. **Output gate** — يقرر يطلع إيه كـ hidden state
	```
	o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
	h_t = o_t ⊙ tanh(c_t)
	```

الـ **cell state** `c_t` هو “ذاكرة طويلة” بتعدّي عبر السلسلة بتعديلات بسيطة، فالتعلّم بيبقى أسهل.

**بالعربي البسيط:** الموديل ممكن يتعلم إن جملة زي "no evidence" بدري في المقال تفضل مؤثرة لما يشوف "as sources confirm" بعدين.

---

### المعمارية بالتفصيل (The Architecture in Detail)

```python
Sequential([
	 # Layer 1: Embedding (word index → 100-dim vector)
	 Embedding(vocab_size=20000, output_dim=100,
				  weights=[glove_matrix], trainable=True),
    
	 # Layer 2: LSTM (sequence of 100-dim vectors → single 128-dim vector)
	 LSTM(units=128, dropout=0.3, recurrent_dropout=0.3),
    
	 # Layer 3: Dense (128-dim → 64-dim, non-linear transformation)
	 Dense(units=64, activation='relu'),
    
	 # Layer 4: Dropout (regularization)
	 Dropout(rate=0.3),
    
	 # Layer 5: Output (64-dim → 1 probability)
	 Dense(units=1, activation='sigmoid')
])
```

**ليه 128 units؟** رقم شائع كبداية. أكبر (256/512) أدق ممكن بس أبطأ وoverfitting أعلى. أصغر (64) أسرع بس ممكن underfit.

**ليه Dense(64) بعد LSTM؟** علشان يعمل non-linear mix للخصايص اللي طلعها الـ LSTM قبل التصنيف.

**ليه ReLU؟** default قوي وسريع وبيقلل مشاكل vanishing gradients في الطبقات.

**ليه Sigmoid في الآخر؟** علشان نطلع احتمال بين 0 و1 مناسب لـ binary crossentropy.

---

### Dropout: التعميم ومنع الـ overfitting

**الفكرة:** أثناء التدريب بنصفر نسبة من النيورونز عشوائيًا.

**ليه؟** يمنع الـ co-adaptation ويجبر الشبكة تتعلم تمثيلات أعمّ.

**`dropout=0.3` vs `recurrent_dropout=0.3`:**
- `dropout`: على مدخلات الـ LSTM
- `recurrent_dropout`: على وصلات الذاكرة (hidden-to-hidden)

---

### Padding: ليه 300 توكن؟ (Sequence Padding: Why 300 Tokens?)

الشبكات محتاجة مدخل ثابت. المقالات أطوالها مختلفة، فبنستخدم padding/truncation:

```python
pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
```

**ليه 300؟** بعد التنضيف المتوسط ~250 توكن، فـ 300 بتغطي أغلب المقالات. الطويل بيتقص (truncation).

**ملحوظة:** قصّ الذيل غالبًا مش مؤذي هنا لأن إشارات الفيك كتير بتبان في العنوان/بداية المقال.

---

### Early Stopping: نقف قبل ما نحفظ الداتا (Preventing Overfitting)

```python
EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
```

لو `val_accuracy` مبقاش يتحسن 3 epochs ورا بعض، بنقف ونرجّع أحسن وزن.

---

### Class Weights في Keras

```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
model.fit(..., class_weight={0: class_weights[0], 1: class_weights[1]})
```

حتى لو الـ imbalance بسيط، ده بيضمن إن gradient signal متوازن.

---

## 8. Evaluation — Measuring What Actually Matters

### ليه الـ Accuracy لوحده مضلل؟ (Why Accuracy Alone Is Misleading)

لو 95% من الداتا Real و5% Fake، موديل بيقول Real دايمًا هياخد 95% Accuracy بس هو عديم القيمة. عشان كده بنبص على مقاييس تانية.

---

### Confusion Matrix

```
					  Predicted REAL  Predicted FAKE
Actual REAL         6,586 (TN)      338 (FP)
Actual FAKE           357 (FN)    5,246 (TP)
```

- **TP:** فيك واتمسك (ده اللي عايزينه يزيد)
- **TN:** ريل واتصنّف صح
- **FP:** خبر ريل اتظلم واتقال عليه فيك (خطر رقابة/اتهام)
- **FN:** خبر فيك اتقال عليه ريل (خطر تضليل يعدّي)

الـ tradeoff بيعتمد على التطبيق:
- **Censorship-averse:** نقلل FP
- **Misinformation-averse:** نقلل FN

---

### Precision / Recall / F1

**Precision (لفئة FAKE):**
```
Precision = TP / (TP + FP) = 5246 / (5246 + 338) = 0.939
```
يعني من اللي قلت عليهم Fake، كام واحد طلع فعلًا Fake.

**Recall (لفئة FAKE):**
```
Recall = TP / (TP + FN) = 5246 / (5246 + 357) = 0.936
```
يعني من كل الفيك الحقيقي، كام واحد اتلقط.

**F1:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
ده harmonic mean وبيعاقب عدم التوازن بين precision وrecall.

---

### Precision-Recall Tradeoff (الـ Threshold)

الـ threshold الافتراضي 0.5. لو قللته (0.3) هتزود الـ recall وتقلل precision. لو علّيته (0.7) العكس.

في الإنتاج، بتختار threshold حسب احتياج البزنس: مراجعة بشرية؟ ولا إزالة تلقائية؟

---

### Cross-Validation للاستقرار

```python
cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1_macro')
# Result: 0.939 ± 0.003
```

**F1 macro:** بيحسب متوسط F1 للكلاسين بنفس الوزن (مناسب للتوازن).

الـ std الصغير بيقول إن الأداء ثابت.

---

## 9. Explainability — Why Did the Model Say That?

قابلية التفسير مهمة جدًا في fake news detection لأن:
1. الصحفي/الفاكت شيكر لازم يفهم ليه المقال اتفلّج
2. الموديلات ممكن تتعلم حاجات سبورياس (زي "reuters" = real) ولازم ده يبان
3. المستخدم محتاج يثق/يعاير الثقة في النتيجة

---

### Explainability في Logistic Regression: أهمية الـ Features الخطية

**الفكرة:** توقع المقال هو مجموع مساهمات الكلمات:

```
score = sum(w_i × tfidf_i for all i)
```

مساهمة كل كلمة = `w_i × tfidf_i` (وزن الكلمة في الموديل × قيمتها في المقال).

```python
def explain_lr_prediction(text, vectorizer, model, top_n=10):
	vec = vectorizer.transform([text])  # Shape: (1, 5000)
	coefs = model.coef_[0]             # Shape: (5000,)
	contributions = vec.toarray()[0] * coefs  # Element-wise product
    
	# Get word names for non-zero features
	feature_names = vectorizer.get_feature_names_out()
	nonzero_idx = vec.nonzero()[1]
    
	word_contributions = [
		(feature_names[i], contributions[i])
		for i in nonzero_idx
	]
	word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
	return word_contributions[:top_n]
```

**ليه ده شغال قوي؟** لأن LR خطي: الشرح ده **بالظبط** نفس الحساب اللي الموديل عمله (مش approximation).

**مثال:**
```
Article: "SHOCKING: Government HIDES Truth About COVID Vaccine Side Effects"
Top FAKE signals: [("shocking", 0.82), ("hides", 0.61), ("truth", 0.45)]
Top REAL signals: [("government", -0.23), ("side", -0.11)]
```

---

### Explainability في LSTM: Perturbation-based analysis

**ليه الشرح الخطي ماينفعش؟** LSTM غير خطي وفيه تفاعلات بين الكلمات، ومفيش وزن واحد لكل كلمة مستقل.

**الطريقة هنا:** نشيل/نغطّي كلمة ونشوف الاحتمال اتغير قد إيه.

```python
def explain_lstm_prediction(text, model, tokenizer, baseline_prob, top_n=10):
	tokens = text.split()
	word_importances = []
    
	for i, word in enumerate(tokens):
		# Mask the word (replace with OOV token)
		masked_tokens = tokens.copy()
		masked_tokens[i] = '<OOV>'
		masked_text = ' '.join(masked_tokens)
        
		# Get prediction without this word
		masked_seq = tokenizer.texts_to_sequences([masked_text])
		masked_padded = pad_sequences(masked_seq, maxlen=300)
		masked_prob = model.predict(masked_padded, verbose=0)[0][0]
        
		# Importance = how much prediction changes when word is removed
		delta = abs(baseline_prob - masked_prob)
		word_importances.append((word, delta))
    
	word_importances.sort(key=lambda x: x[1], reverse=True)
	return word_importances[:top_n]
```

**الفكرة:** لو إزالة الكلمة غيّرت الاحتمال جامد → الكلمة كانت مؤثرة.

**الحدود:** ده approximation، وبيحسب تأثير الكلمة لوحدها، مع إن الكلمات بتتفاعل. مثال: معنى "not" بيبان مع اللي حواليه.

**بدائل أحسن (مش متطبقة هنا):**

| Method | Description | Complexity |
|--------|-------------|------------|
| SHAP | game-theoretic وبيحسب تفاعلات | High |
| LIME | local linear approximation | Medium |
| Gradient × Input | backprop للـ input | Medium |
| Attention visualization | لو موديل attention-based | Low |

---

## 10. Model Comparison and Lessons Learned

### النتائج جنب بعض (Results Side by Side)

| Metric | Logistic Regression + TF-IDF | LSTM + GloVe |
|--------|------------------------------|--------------|
| Accuracy | **94.60%** | 94.52% |
| Precision (FAKE) | 94.28% | 94.28% |
| Recall (FAKE) | 93.61% | **94.24%** |
| F1-score | 93.95% | **94.26%** |
| Training time | ~5 ثواني | ~1,200 ثانية |
| Inference speed | ~1ms/مقال | ~20ms/مقال |
| Interpretability | Exact | Approximate |
| Context awareness | None | Sequential |

### ليه قريبين قوي كده؟ (Why Are They So Similar?)

ده أهم درس في المشروع.

**الأداء شبه بعض لأن المشكلة Lexical أكتر ما هي Sequential.** الإشارات الأساسية:
1. اختيار الكلمات ("shocking", "conspiracy", "deep state")
2. attribution patterns ("said", "according to")
3. مصطلحات الدومين

كل ده TF-IDF بيشيله كويس. الـ LSTM مش بيكسب كتير لأن ترتيب الكلمات مش هو السيجنال المسيطر في الداتا دي.

### إمتى LSTM/Transformers يتفوّقوا على TF-IDF + LR؟

1. **Negation:** "did NOT endorse" vs "did endorse" (bag-of-words ممكن يتلخبط)
2. **Sarcasm/Irony:** أسلوب ساخر محتاج سياق
3. **Cross-sentence reasoning:** استدلال عبر جمل
4. **Entity disambiguation:** "the president" مين؟

في الداتا دي الأنماط دي مش هي الغالبة.

### الدرس الأكبر: ابدأ بـ Baseline بسيط

التسلسل العملي في NLP غالبًا:
1. TF-IDF + Logistic Regression
2. لو مش كفاية: LSTM/CNN + embeddings
3. لو لسه: Transformers (BERT/RoBERTa)
4. متكبّرش الموديل إلا لو محتاج

المشروع ده بيورّي ده بوضوح: LSTM أغلى ~240× في الوقت، وتحسينه هامشي.

---

## 11. The Web UI — Bringing It All Together

### المعمارية (Architecture)

تطبيق Streamlit في (`ui.py`) بيحمّل الموديلات المتدرّبة وبيقدّم واجهة تفاعلية للتوقع.

**استراتيجية تحميل الموديلات (Model loading strategy):**

```python
# Logistic Regression (always available)
lr_model = joblib.load('outputs/models/logistic_model.pkl')
vectorizer = joblib.load('outputs/models/tfidf_vectorizer.pkl')

# LSTM (graceful degradation if unavailable)
try:
	lstm_model = tf.keras.models.load_model('outputs/models/lstm_model.h5')
	tokenizer = joblib.load('outputs/models/tokenizer.pkl')
	lstm_available = True
except Exception:
	lstm_available = False
```

**Graceful degradation** فكرة سوفتوير مهمة: لو الـ LSTM فشل يتحمّل لأي سبب (نسخة TensorFlow، ملف ناقص)، الـ UI لسه شغال بـ Logistic Regression بدل ما ينهار.

### شرط تماثل الـ preprocessing (The Preprocessing Symmetry Requirement)

لازم `clean_text` في الـ UI تبقى **نفسها حرفيًا** اللي اتعمل بيها training في الـ notebooks.

**ليه؟** الـ TF-IDF vectorizer اتعمله fit على ناتج التنضيف ده. لو الـ UI بيعمل تنضيف مختلف حتى لو بسيط، هتحصل مشكلة **training-serving skew** (فرق بين اللي اتدرّب عليه واللي بيتقدّم له في الإنتاج)، وده يقلل الأداء ويزوّد OOV tokens.

### عرض الموديلين والتنبيه عند الاختلاف (Two-Model Disagreement)

الـ UI بيعرض توقع الموديلين وبيحط تحذير لو حصل اختلاف:

```python
if lr_pred != lstm_pred:
	st.warning("Models disagree — treat this prediction with caution.")
```

ده مؤشر بسيط بس مفيد لعدم اليقين: لما موديلين مختلفين في المعمارية يختلفوا، غالبًا المثال قريب من الـ decision boundary.

---

## 12. What Would Professionals Do Next?

المشروع ده أساس قوي. حاجات “محترفين” عادة يضيفوها:

### 1. استخدام Transformer (BERT / RoBERTa)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
	"roberta-base", num_labels=2
)
```

**ليه؟** Transformers متدرّبة مسبقًا على بيانات ضخمة وبتفهم سياق/نفي/نُسَك أدق من LSTM. غالبًا توصل 97–99% على داتا شبه دي.

**التكلفة:** موديل أكبر بكتير، محتاج GPU غالبًا، وlatency أعلى.

### 2. معالجة تحيّز الداتا (Dataset Bias)

الداتا دي ممكن تكون بتلابل حسب المصدر. لازم تختبر على cross-source data وتشوف هل الموديل “بيحفظ” مصادر بدل محتوى.

### 3. Temporal validation

قسّم التدريب/الاختبار حسب التاريخ (train قديم، test أحدث) علشان تشوف التعميم مع تغيّر اللغة والأحداث.

### 4. Calibration

الـ probabilities بتاعة الكلاسيـفاير مش دايمًا calibrated. استخدم temperature scaling أو Platt scaling لو عايز احتمالات “مُعتمدة”.

### 5. Active learning

ركّز labeling على الأمثلة اللي الموديل مش واثق فيها (قريبة من 0.5) بدل labeling عشوائي.

### 6. Multi-class detection

بدل binary، خليك multi-class (Satire / Misleading / Propaganda / Fabricated) علشان التدخّل يختلف حسب النوع.

### 7. اعتبارات الإنتاج (Production Considerations)

| Concern | Solution |
|---------|----------|
| Model staleness | retrain دوري |
| Adversarial attacks | مراقبة distribution shift |
| API serving | FastAPI + MLflow registry |
| Latency | TF-IDF + LR وربما ONNX |
| Monitoring | تتبّع توزيعات الثقة مع الوقت |

---

## Summary: خريطة المعرفة في المشروع (The NLP Knowledge Map)

```
Text Input
	↓
[Preprocessing]
 Lowercase → URL removal → Punctuation removal →
 Tokenization → Stopword removal → Lemmatization
	↓
[Feature Extraction]
  ┌─────────────────┐       ┌──────────────────────┐
  │  TF-IDF (Sparse) │       │ GloVe → LSTM (Dense) │
  │  5000 features   │       │ 20000 vocab, 100-dim  │
  │  Bigrams         │       │ Sequential context    │
  └─────────────────┘       └──────────────────────┘
	↓                              ↓
[Classification]           [Classification]
Logistic Regression         LSTM → Dense → Sigmoid
Linear decision boundary    Non-linear decision boundary
	↓                              ↓
[Evaluation]               [Evaluation]
94.60% accuracy             94.52% accuracy
0.939 F1                    0.943 F1
	↓                              ↓
[Explainability]           [Explainability]
Coefficient × TF-IDF        Perturbation analysis
(Exact)                     (Approximate)
	↓                              ↓
			  [Web UI]
		Side-by-side predictions
		Confidence scores
		Top contributing words
```

**الخلاصة الأساسية:** في fake news detection (على الداتا دي)، “الكلمات المستخدمة” أقوى سيجنال من “ترتيب الكلمات”. عشان كده البداية بـ baseline بسيط (TF-IDF + LR) بتبقى منطقية جدًا، والتعقيد بييجي بس لو محتاج.

---

*الملف ده ترجمة للعامية المصرية من الدليل الأصلي، مع الحفاظ على الكود/الأسماء التقنية زي ما هي قدر الإمكان.*
