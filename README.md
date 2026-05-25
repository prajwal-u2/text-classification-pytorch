# Tweet Disaster Classification (NLP + Transfer Learning)

Binary text classification on tweets: predict whether a tweet describes a **real disaster** or not. Built as a hands-on NLP learning project using **PyTorch** and **Hugging Face Transformers**, with **fine-tuning** of a pre-trained **DistilBERT** model.

> **Personal study reference:** [docs/learning.md](docs/learning.md) — concepts covered in this project, explained for later review.

---

## Problem

Social media posts about disasters are noisy and ambiguous. A keyword like “fire” might refer to a real emergency or casual slang. The goal is to learn a model that understands **context in short text**, not just individual words.

| Label | Meaning |
|-------|---------|
| `1` | Disaster-related tweet |
| `0` | Not disaster-related |

**Dataset:** ~7,600 labeled training tweets and ~3,200 unlabeled test tweets ([`data/train.csv`](data/train.csv), [`data/test.csv`](data/test.csv)). Same structure as the popular [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) Kaggle competition.

---

## What I Built

End-to-end NLP pipeline in [`src/Tweet_Classification.ipynb`](src/Tweet_Classification.ipynb):

1. **Exploratory analysis** — class distribution, word clouds (before/after cleaning)
2. **Text preprocessing** — custom cleaning for Twitter-style text
3. **Tokenization** — DistilBERT tokenizer (`input_ids`, `attention_mask`)
4. **Model** — `DistilBertForSequenceClassification` loaded from `distilbert-base-uncased`
5. **Fine-tuning** — train on labeled tweets with PyTorch training loop
6. **Evaluation** — validation metrics, classification report, confusion matrix
7. **Inference** — predictions and probabilities on the held-out test set

The classifier head is trained from scratch; the DistilBERT backbone is **fine-tuned** (all parameters updated, not frozen).

---

## Approach (How It Works)

```
Raw tweets → clean_text() → DistilBERT tokenizer → TensorDataset / DataLoader
    → DistilBERT + classification head → AdamW + cross-entropy loss
    → 3 epochs → save best checkpoint (val F1) → evaluate → test predictions
```

### Text cleaning

Tweets are normalized before tokenization: lowercase, remove URLs, mentions, HTML, digits, and extra whitespace; strip hashtag symbols while keeping words.

### Model & transfer learning

- **Base model:** [DistilBERT](https://huggingface.co/distilbert-base-uncased) — a lighter, faster variant of BERT, pre-trained on large English corpora.
- **Task adaptation:** `DistilBertForSequenceClassification` with `num_labels=2`; the pre-trained encoder provides language understanding, and fine-tuning adapts it to disaster vs non-disaster language.
- **Why transformers:** Self-attention captures context in short, informal text better than bag-of-words or simple RNN baselines for this kind of task.

### Training setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (`lr=2e-5`) |
| Batch size | 128 |
| Max sequence length | 128 tokens |
| Epochs | 3 |
| Train/val split | 90% / 10%, stratified |
| Device | GPU when available (Colab T4), else CPU |
| Checkpoint | Best model by validation **F1** → `best_model.pt` (saved in notebook working directory) |

### Reproducibility

Fixed random seeds (`SEED=42`) for PyTorch, NumPy, and Python; deterministic cuDNN when on GPU.

---

## Results (Validation Set)

After fine-tuning, on the held-out validation split (~762 tweets):

| Metric | Score |
|--------|-------|
| Accuracy | **0.83** |
| Macro F1 | **0.82** |
| Best validation F1 (during training) | **0.82** |

Per-class (approximate from final evaluation):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Not Disaster | 0.85 | 0.84 | 0.85 |
| Disaster | 0.79 | 0.80 | 0.80 |

Training was run in Google Colab with GPU. Metrics can vary slightly by hardware and library versions.

---

## Tech Stack

| Area | Tools |
|------|-------|
| Deep learning | PyTorch, Hugging Face `transformers` |
| Model | DistilBERT (`distilbert-base-uncased`) |
| Data & ML | pandas, NumPy, scikit-learn |
| NLP utilities | NLTK (stopwords), WordCloud |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter / Google Colab |

---

## Skills Demonstrated

- **NLP pipeline design** — preprocessing → tokenization → model → evaluation → inference
- **Transfer learning & fine-tuning** — loading pre-trained weights and adapting them to a downstream classification task
- **Transformer models** — BERT-family tokenization, attention masks, sequence classification heads
- **PyTorch fundamentals** — `DataLoader`, training/eval loops, device placement, checkpointing
- **ML best practices** — stratified splits, class-imbalance-aware metric (F1), reproducibility, validation-driven model selection
- **Model evaluation** — precision, recall, F1, confusion matrix, error analysis mindset

---

## Project Structure

```
.
├── README.md
├── docs/
│   └── learning.md              # Study notes — NLP concepts learned
├── src/
│   └── Tweet_Classification.ipynb
├── data/
│   ├── train.csv
│   └── test.csv
└── best_model.pt                # Generated after training (optional; path depends on run cwd)
```

---

## How to Run

1. Clone the repo. Data files are in [`data/`](data/).
2. Open [`src/Tweet_Classification.ipynb`](src/Tweet_Classification.ipynb) in Jupyter or [Google Colab](https://colab.research.google.com/).
3. Run from `src/` (notebook loads data via `../data/train.csv` and `../data/test.csv`).

4. Install dependencies (notebook installs as needed):

   ```bash
   pip install pandas numpy matplotlib seaborn torch scikit-learn transformers nltk wordcloud
   ```

5. Run all cells top to bottom. GPU is recommended for fine-tuning (several minutes on T4 vs much longer on CPU).
6. Optional: download NLTK stopwords when prompted by the word-cloud cells.

---

## Notebook Sections

| # | Section |
|---|---------|
| 1 | Setup & reproducibility |
| 2 | Data loading & exploration |
| 3–5 | Word clouds, text cleaning, cleaned EDA |
| 6 | DistilBERT tokenization |
| 7 | Train/val split & DataLoaders |
| 8 | Load pre-trained model |
| 9 | Training & evaluation functions |
| 10 | Fine-tuning loop |
| 11 | Final validation metrics & confusion matrix |
| 12 | Test set predictions |

---

## Learnings & Next Steps

This project was built to **learn core NLP ideas** in practice: text cleaning, subword tokenization, pre-trained language models, and fine-tuning. Possible extensions (not implemented here): class-weighted loss for imbalance, learning-rate scheduling, early stopping, k-fold CV, or comparing against a non-transformer baseline (e.g. TF-IDF + logistic regression).

For detailed concept explanations, see **[docs/learning.md](docs/learning.md)**.
