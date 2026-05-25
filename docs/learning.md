# NLP Concepts — Study Reference

Personal notes from the **Tweet Disaster Classification** project. Use this doc to revisit what each idea means and how it showed up in [`src/Tweet_Classification.ipynb`](../src/Tweet_Classification.ipynb).

---

## Table of Contents

1. [What is NLP?](#1-what-is-nlp)
2. [The Task: Text Classification](#2-the-task-text-classification)
3. [Raw Text vs Clean Text](#3-raw-text-vs-clean-text)
4. [Text Preprocessing](#4-text-preprocessing)
5. [Exploratory Analysis (EDA)](#5-exploratory-analysis-eda)
6. [From Words to Numbers](#6-from-words-to-numbers)
7. [Tokenization (Subword)](#7-tokenization-subword)
8. [Attention Masks](#8-attention-masks)
9. [Embeddings & Context](#9-embeddings--context)
10. [Transformers & BERT Family](#10-transformers--bert-family)
11. [DistilBERT](#11-distilbert)
12. [Transfer Learning](#12-transfer-learning)
13. [Fine-Tuning](#13-fine-tuning)
14. [Sequence Classification Head](#14-sequence-classification-head)
15. [Loss Function](#15-loss-function)
16. [Training Loop (PyTorch)](#16-training-loop-pytorch)
17. [Optimizer: AdamW](#17-optimizer-adamw)
18. [Batches & DataLoaders](#18-batches--dataloaders)
19. [Train / Validation / Test](#19-train--validation--test)
20. [Evaluation Metrics](#20-evaluation-metrics)
21. [Confusion Matrix](#21-confusion-matrix)
22. [Inference on New Data](#22-inference-on-new-data)
23. [Reproducibility](#23-reproducibility)
24. [Common Pitfalls](#24-common-pitfalls)
25. [Concept Map](#25-concept-map)
26. [Glossary](#26-glossary)

---

## 1. What is NLP?

**Natural Language Processing (NLP)** is the branch of ML/AI that works with human language — text or speech.

Typical tasks:

| Task | Example |
|------|---------|
| Classification | Spam vs not spam; disaster vs not |
| Named entity recognition | Find person/place names in text |
| Translation | English → Spanish |
| Summarization | Long article → short summary |
| Question answering | Answer from a document |

This project: **classification** (assign one label per tweet).

---

## 2. The Task: Text Classification

**Text classification** = given a piece of text, predict a category (label).

Here: **binary classification** — exactly two classes.

- **Input:** tweet string  
- **Output:** `0` (not disaster) or `1` (disaster)

Challenges in this dataset:

- Short text (limited context)
- Informal language, slang, typos
- Ambiguous words (“fire”, “crash” can be literal or figurative)
- Noise: URLs, @mentions, hashtags, HTML

The model must use **context**, not single keywords alone.

---

## 3. Raw Text vs Clean Text

**Raw text** = exactly as stored in the CSV (`text` column).

**Clean text** = normalized version (`clean_text`) after a `clean_text()` function.

Why clean?

- Reduces noise the model might overfit on (URLs, user handles)
- Standardizes format (lowercase, single spaces)
- Keeps **semantic words** that matter for disasters

What we did **not** do (worth knowing):

- Stemming/lemmatization (DistilBERT tokenizer handles word forms via subwords)
- Aggressive stopword removal before modeling (stopwords were used for **word clouds**, not for the model)

---

## 4. Text Preprocessing

Preprocessing = transforming raw strings into a form suitable for the model.

### Steps in this project

| Step | What it does | Why |
|------|----------------|-----|
| Lowercase | `text.lower()` | Consistency |
| Remove URLs | `https://...`, `www....` | Not predictive of disaster meaning |
| Remove @mentions | `@user` | User IDs don’t generalize |
| Strip `#` | Keep word, drop symbol | Hashtag words often matter |
| Remove `rt` | Retweet marker | Noise |
| Remove HTML | `<tag>` | Artifacts from scraping |
| Remove special chars & digits | punctuation, numbers | Simplifies surface form |
| Collapse whitespace | single spaces | Cleaner input |

### Regex (`re.sub`)

Pattern-based find-and-replace. Example: `r'@\w+'` matches `@` followed by word characters.

### Trade-off

Heavy cleaning can **remove useful signal** (e.g. “911”, “magnitude 7”). For transformers, light-to-moderate cleaning is common; the model still sees subword tokens for remaining words.

---

## 5. Exploratory Analysis (EDA)

EDA = understand data **before** heavy modeling.

### What we did

- **Shape & head:** row counts, column names, sample tweets
- **Class balance:** count of disaster vs non-disaster (`target`)
- **Word clouds:** visual frequency of words per class (before and after cleaning)

### Why word clouds?

Quick intuition: which words dominate each class? Do disaster tweets cluster around “fire”, “evacuation”, “earthquake”?

### NLTK stopwords

Common words (“the”, “is”, “and”) filtered in word clouds so rare/ meaningful words stand out. **Not** applied to DistilBERT inputs in this notebook.

---

## 6. From Words to Numbers

Neural networks need **numbers**, not strings.

Pipeline:

```
"forest fire near canada"  →  tokenizer  →  [101, 4946, 2548, ...]  →  model  →  logits  →  class
```

Two eras of approaches:

| Approach | Idea | This project |
|----------|------|----------------|
| Classical | TF-IDF / bag-of-words + logistic regression, SVM | Not used |
| Neural / Transformer | Pre-trained LM + fine-tune | **Used** |

---

## 7. Tokenization (Subword)

**Tokenization** splits text into **tokens** (pieces the model understands).

### Word-level (old)

`"playing"` → one token. Rare words → unknown (`<UNK>`).

### Subword (BERT / DistilBERT)

Uses **WordPiece**: frequent words stay whole; rare words split.

Examples:

- `"playing"` might stay one piece
- `"epidemiological"` → `["epidem", "##iological"]`

`##` means “continuation of previous token.”

### `DistilBertTokenizer`

- Loaded from `distilbert-base-uncased` (same vocabulary as training)
- `convert_tokens_to_ids` maps tokens → integer IDs
- Special tokens: `[CLS]` (start), `[SEP]` (separator), `` (padding)

### `max_length=128`

Each tweet is truncated or padded to 128 tokens. Longer tweets lose tail words; shorter tweets get ``.

### `padding='max_length'`

Every sequence in a batch has the same length (128) for efficient tensor operations.

---

## 8. Attention Masks

Transformers use **self-attention**: each token can “look at” other tokens.

**Padding tokens** should not be attended to — they’re filler.

**Attention mask:**

- `1` = real token (attend)
- `0` = padding (ignore)

Passed to the model as `attention_mask` alongside `input_ids`.

---

## 9. Embeddings & Context

### Embedding

A **dense vector** representing a token (or word). Similar meaning → often similar vectors in embedding space.

### Contextual embeddings (BERT-style)

Unlike static Word2Vec, **the same word gets different vectors depending on sentence context**.

Example: “bank” in “river bank” vs “bank account” — different representations after the transformer layers.

That’s why transformers work well for ambiguous short text.

---

## 10. Transformers & BERT Family

### Transformer (architecture)

- **Encoder** stacks: self-attention + feed-forward layers
- Processes full sequence in parallel (vs RNN step-by-step)
- Captures long-range dependencies in one pass

### BERT (Bidirectional Encoder Representations from Transformers)

- Pre-trained on masked language modeling + next sentence prediction
- **Bidirectional:** sees left and right context (good for classification)
- `[CLS]` token at start — its final hidden state is often used for **sentence-level** tasks

### BERT for classification

```
[CLS] tweet tokens [SEP]  →  encoder  →  [CLS] vector  →  linear layer  →  2 logits
```

---

## 11. DistilBERT

**DistilBERT** = distilled (compressed) BERT:

- ~40% smaller, ~60% faster
- Keeps much of BERT’s performance
- Same tokenizer family (`uncased` = lowercase English)

Good for learning and smaller GPUs (Colab T4).

Model class used: `DistilBertForSequenceClassification`.

---

## 12. Transfer Learning

**Transfer learning** = reuse knowledge from one task/domain for another.

Here:

| Source task | Target task |
|-------------|-------------|
| General English (books, Wikipedia, etc.) | Disaster tweet classification |

**What transfers:** patterns of grammar, semantics, common phrases, context.

**What doesn’t transfer automatically:** label boundary (disaster vs joke about fire) — that’s learned during fine-tuning.

Without transfer learning you’d need huge labeled tweet data and long training from random init.

---

## 13. Fine-Tuning

**Fine-tuning** = continue training a pre-trained model on **your labeled data**.

### What happens at load time

```python
DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
```

- Encoder weights: **loaded** from Hugging Face
- Classifier head (`pre_classifier`, `classifier`): often **randomly initialized** (reported as “MISSING” in load log — expected)

### What we trained

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

`model.parameters()` = **all** weights (encoder + head). Full fine-tune, not frozen backbone.

### Hyperparameters that matter

| Param | Our value | Note |
|-------|-----------|------|
| Learning rate | `2e-5` | Small — standard for BERT fine-tune |
| Epochs | 3 | More can overfit on small data |
| Batch size | 128 | Limited by GPU memory |

### `model.train()` vs `model.eval()`

- **Train:** dropout on, gradients computed, weights updated
- **Eval:** dropout off, no gradients (`torch.no_grad()`), stable metrics

---

## 14. Sequence Classification Head

Hugging Face wraps:

1. DistilBERT encoder
2. **Pooling** of representation (uses `[CLS]`-style output)
3. **Linear layer** → `num_labels` logits

For binary: 2 logits → softmax → probabilities for class 0 and 1.

`num_labels=2` tells the library to build a 2-way classifier.

---

## 15. Loss Function

**Cross-entropy loss** for multi-class / binary classification.

When you pass `labels=targets` into the model forward pass, Hugging Face computes loss internally:

```python
outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
loss = outputs.loss
```

**Intuition:** penalize the model when it assigns low probability to the correct class.

`loss.backward()` → gradients  
`optimizer.step()` → update weights

---

## 16. Training Loop (PyTorch)

Manual loop (educational — production often uses `Trainer` API).

### One epoch (`train_epoch`)

1. `model.train()`
2. For each batch:
   - Move `input_ids`, `attention_mask`, `targets` to device
   - Forward → loss
   - `optimizer.zero_grad()`
   - `loss.backward()`
   - `optimizer.step()`
   - Track accuracy
3. Return average loss & accuracy

### Evaluation (`evaluate`)

1. `model.eval()`
2. `torch.no_grad()` — no gradient storage
3. Forward without updating weights
4. Collect preds, compute F1

### Checkpointing

Save `state_dict()` when validation F1 improves → `best_model.pt`.

---

## 17. Optimizer: AdamW

**AdamW** = Adam with decoupled weight decay (standard for transformers).

- Adaptive per-parameter learning rates
- Works well with small LR (`2e-5`) on pre-trained models

Why not plain SGD? Transformers are sensitive; Adam-family optimizers are the default in papers and HF examples.

---

## 18. Batches & DataLoaders

### Why batches?

- GPU processes many examples in parallel
- Stable gradient estimates

### `TensorDataset`

Bundles tensors: `(input_ids, attention_mask, targets)` per sample.

### `DataLoader`

- `batch_size=128`
- `shuffle=True` on train (different order each epoch)
- No shuffle on val/test (order doesn’t matter for metrics)

### Shapes (typical batch)

```
input_ids:      [128, 128]   # batch_size × seq_len
attention_mask: [128, 128]
targets:        [128]
```

---

## 19. Train / Validation / Test

| Split | Purpose | In project |
|-------|---------|------------|
| **Train** | Learn weights | 90% of labeled data |
| **Validation** | Tune/compare models, pick checkpoint | 10%, stratified |
| **Test** | Final generalization (labels hidden in competition) | `test.csv`, predict only |

### Stratified split

`stratify=train_df['target']` keeps the same % of disaster vs non-disaster in train and val.

Important when classes are imbalanced.

### Data leakage caution

- Fit tokenizer vocabulary: **pre-defined** by pre-training (not fit on our labels — good)
- Don’t tune on test labels
- Clean text the same way for train and test

---

## 20. Evaluation Metrics

### Accuracy

Fraction of correct predictions. Can mislead if classes are imbalanced.

### Precision (per class)

Of everything predicted as class X, how many were truly X?

`TP / (TP + FP)`

### Recall (per class)

Of all true class X, how many did we find?

`TP / (TP + FN)`

### F1-score

Harmonic mean of precision and recall. Balances both.

**Macro F1:** average F1 across classes (treats classes equally).  
**Weighted F1:** weighted by support (class frequency).

We used **validation F1** to select the best epoch — reasonable for imbalanced-ish binary tasks.

### Classification report

sklearn `classification_report` — precision, recall, F1, support per class.

---

## 21. Confusion Matrix

Table of **true label vs predicted label**.

```
                 Predicted
                 0      1
Actual  0      TN     FP
        1      FN     TP
```

Helps see **which mistakes** happen:

- Many FP → model cries “disaster” too often
- Many FN → model misses real disasters

Visualized with seaborn heatmap in the notebook.

---

## 22. Inference on New Data

Steps for `test.csv`:

1. `clean_text` on tweets
2. `tokenize_text` → `input_ids`, `attention_mask`
3. `DataLoader` (no labels)
4. `model.eval()` + `torch.no_grad()`
5. `argmax` on logits → `predicted_target`
6. Optional: softmax probabilities → `disaster_probability`

No weight updates during inference.

---

## 23. Reproducibility

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# + CUDA seeds, cudnn deterministic
```

Same seed + same environment → similar (not always identical) results.

Useful for debugging and comparing experiments.

---

## 24. Common Pitfalls

| Pitfall | What to remember |
|---------|------------------|
| Training on test data | Test is for final eval only |
| Wrong tokenizer | Must match model (`distilbert-base-uncased`) |
| Too high LR | Destroys pre-trained weights; use ~1e-5 to 5e-5 |
| Ignoring class imbalance | Watch F1, not only accuracy |
| Max length too short | Long tweets truncated — may lose signal |
| Forgetting `attention_mask` | Model attends to padding |
| Eval mode left off | Dropout randomness skews inference |

---

## 25. Concept Map

```
                    ┌─────────────────┐
                    │   Raw tweets    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Text cleaning  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Subword tokens │
                    │  + attn mask    │
                    └────────┬────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │     DistilBERT (pre-trained encoder)     │
         │         transfer learning                │
         └───────────────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Classifier head │
                    │  (fine-tuned)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Train loop    Validation      Test preds
         (loss/acc)    (F1, report)    (submission)
```

---

## 26. Glossary

| Term | Short definition |
|------|------------------|
| **NLP** | ML on human language |
| **Token** | Subword unit in model vocabulary |
| **Tokenizer** | Text → token IDs |
| **Embedding** | Vector representation of a token |
| **Transformer** | Attention-based neural architecture |
| **BERT** | Bidirectional pre-trained transformer encoder |
| **DistilBERT** | Smaller/faster distilled BERT |
| **Hugging Face** | Library/host for pre-trained models |
| **Transfer learning** | Reuse pre-trained model on new task |
| **Fine-tuning** | Train (some or all) pre-trained weights on new labels |
| **Logits** | Raw scores before softmax |
| **Softmax** | Converts logits to probabilities |
| **Epoch** | One full pass over training data |
| **Batch** | Subset of data processed together |
| **Stratified split** | Split preserving class proportions |
| **F1** | Balance of precision and recall |
| **Confusion matrix** | Counts of prediction vs truth |
| **Overfitting** | Great on train, worse on new data |
| **`[CLS]`** | Special token; sentence summary often taken from it |
| **``** | Padding token to fixed length |

---

## Further Reading (optional)

- [Hugging Face — Fine-tuning](https://huggingface.co/docs/transformers/training)
- [DistilBERT paper](https://arxiv.org/abs/1910.01108)
- [BERT paper](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (original Transformer)

---

*Last updated to match [`src/Tweet_Classification.ipynb`](../src/Tweet_Classification.ipynb) pipeline (DistilBERT, PyTorch, 3 epochs, stratified 90/10 split).*
