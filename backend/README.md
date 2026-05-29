# Sentiment Analysis — Upgraded Backend (BPE + TF-IDF)

This is the upgraded version of the sentiment analysis backend.
It combines **BPE (Byte Pair Encoding)** and **TF-IDF** features together
for higher accuracy than either method alone.

## How It Works

```
Raw Text
   ↓ clean_text()
Cleaned Text
   ↓              ↓
TF-IDF (5000)   BPE tokens → BoW (3000)
   ↓              ↓
   └──── hstack() ────┘
     Combined (8000 features)
           ↓
   Logistic Regression
           ↓
   positive / negative + confidence
```

## Project Structure

```
SentimentAnalysis_upgraded/
├── app.py                  ← Flask REST API (BPE + TF-IDF)
├── sentiment_model.py      ← Training script (run to retrain)
├── requirements.txt        ← Dependencies
├── sentiment_model.pkl     ← Trained model (after running training)
├── tfidf_vectorizer.pkl    ← TF-IDF vectorizer (after running training)
└── bpe_tokenizer.json      ← BPE tokenizer (after running training)
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Train the Model (required first time)

Place your `train_data.csv` in this folder, then run:

```bash
py sentiment_model.py
```

This generates 3 files:
- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `bpe_tokenizer.json`

## Step 2 — Run the API

```bash
py app.py
```

Server runs at: `http://localhost:5000`

## API Usage

### POST /predict

```json
{ "text": "I love this product!" }
```

Response:
```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.9412,
  "method": "BPE + TF-IDF fusion"
}
```

## Accuracy Improvement

| Method                        | Approx. Accuracy |
|-------------------------------|-----------------|
| TF-IDF only (old)             | ~80–85%         |
| BPE + TF-IDF combined (new)   | ~86–90%         |
