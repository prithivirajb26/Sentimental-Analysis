"""
Upgraded Sentiment Analysis Model
===================================
Strategy: BPE + TF-IDF Combined Feature Fusion
- BPE (Byte Pair Encoding)  → captures subword patterns, handles rare/unknown words
- TF-IDF                    → captures word importance scores
- Both features are merged  → fed into Logistic Regression
- Negation handling with filler word skip for better accuracy
"""

import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("train_data.csv")
print("📂 Columns:", df.columns.tolist())
print(df.head())

text_col   = "sentence"
target_col = "sentiment"

# -----------------------------------
# Negation Words
# -----------------------------------
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor",
    "hardly", "barely", "scarcely", "without",
    "nobody", "nothing", "nowhere", "none",
    "isnt", "wasnt", "doesnt", "didnt", "wont",
    "cant", "couldnt", "shouldnt", "wouldnt",
    "havent", "hasnt", "hadnt", "arent", "werent"
}

# Filler words — skip these when looking for the word to flip
# Example: "not a bad boy" → skip "a", flip "bad" → "NOT_bad"
FILLER_WORDS = {
    "a", "an", "the", "very", "really", "so",
    "such", "quite", "rather", "too", "just",
    "that", "this", "my", "your", "his", "her",
    "our", "their", "its", "any", "some", "much",
    "more", "most", "even", "only", "ever"
}

# -----------------------------------
# Negation Handler (with filler skip)
# -----------------------------------
def handle_negation(text):
    """
    Flips the first meaningful word after a negation word.
    Skips filler words like a, an, the, very, really etc.

    Examples:
      "not bad"           → "not NOT_bad"
      "not a bad boy"     → "not a NOT_bad boy"
      "never a good idea" → "never a NOT_good idea"
      "not very happy"    → "not very NOT_happy"
      "i am not a bad boy"→ "i am not a NOT_bad boy"  ✅
    """
    words = text.split()
    result = []
    negate = False

    for word in words:
        clean_word = re.sub(r"[^a-z]", "", word)

        if clean_word in NEGATION_WORDS:
            negate = True
            result.append(word)

        elif negate and clean_word in FILLER_WORDS:
            # Filler word — keep negate ON, don't flip yet
            result.append(word)

        elif negate and clean_word:
            # Real meaningful word — flip it
            result.append("NOT_" + clean_word)
            negate = False

        else:
            result.append(word)

    return " ".join(result)

# -----------------------------------
# Text Cleaning
# -----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    # Expand contractions BEFORE removing punctuation
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Apply improved negation handling AFTER cleaning
    text = handle_negation(text)
    return text

# Apply cleaning to entire dataset
df["clean_text"] = df[text_col].astype(str).apply(clean_text)
print("\n🧹 Cleaned text samples:")
print(df[["clean_text", target_col]].head())

# Quick negation test to verify fix is working
print("\n🔍 Negation Handling Test:")
test_cases = [
    "it is not bad",
    "i am not a bad boy",
    "not very happy",
    "never a good idea",
    "not really great",
    "i do not like this",
]
for tc in test_cases:
    result = handle_negation(tc)
    print(f"   '{tc}' → '{result}'")

# -----------------------------------
# METHOD 1 — TF-IDF Vectorization
# -----------------------------------
print("\n📐 Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df["clean_text"])
print("   TF-IDF shape:", X_tfidf.shape)

# -----------------------------------
# METHOD 2 — BPE Tokenization → Feature Matrix
# -----------------------------------
print("\n🔠 Training BPE tokenizer...")

def create_bpe_tokenizer(texts):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(
        vocab_size=3000,
        special_tokens=["[PAD]", "[UNK]"]
    )
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer

bpe_tokenizer = create_bpe_tokenizer(df["clean_text"].tolist())
print("   BPE vocab size:", bpe_tokenizer.get_vocab_size())

def bpe_to_bow(texts, tokenizer, vocab_size=3000):
    rows, cols, data = [], [], []
    for i, text in enumerate(texts):
        encoding = tokenizer.encode(str(text))
        token_ids = encoding.ids
        for tid in token_ids:
            if tid < vocab_size:
                rows.append(i)
                cols.append(tid)
                data.append(1)
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(texts), vocab_size)
    )
    return matrix

print("   Building BPE feature matrix...")
X_bpe = bpe_to_bow(df["clean_text"].tolist(), bpe_tokenizer, vocab_size=3000)
print("   BPE matrix shape:", X_bpe.shape)

# -----------------------------------
# FUSION — Combine TF-IDF + BPE
# -----------------------------------
print("\n🔗 Fusing TF-IDF + BPE features...")
X_combined = hstack([X_tfidf, X_bpe])
print("   Combined feature shape:", X_combined.shape)

y = df[target_col]

# -----------------------------------
# Train / Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
print("\n📊 Training shape:", X_train.shape)
print("📊 Testing shape :", X_test.shape)

# -----------------------------------
# Model Training — Logistic Regression
# -----------------------------------
print("\n🚀 Training model on combined features...")
model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Training complete.")

# -----------------------------------
# Evaluation
# -----------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n📋 Evaluation Results:")
print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# Quick Predictions Test
# -----------------------------------
def predict_sentiment(text, model, tfidf, bpe_tokenizer):
    cleaned    = clean_text(text)
    tfidf_vec  = tfidf.transform([cleaned])
    bpe_vec    = bpe_to_bow([cleaned], bpe_tokenizer, vocab_size=3000)
    combined   = hstack([tfidf_vec, bpe_vec])
    pred       = model.predict(combined)[0]
    proba      = model.predict_proba(combined)[0]
    sentiment  = "Positive 😊" if pred in [1, 4] else "Negative 😞"
    confidence = round(float(max(proba)), 4)
    return sentiment, confidence

test_texts = [
    "I absolutely loved this product, it was amazing!",
    "This is the worst experience I have ever had.",
    "Not bad, could be better though.",
    "I am not a bad boy.",
    "It is not bad.",
    "She is not a good person.",
    "Not very happy with this.",
    "Never a dull moment.",
]

print("\n💬 Quick Predictions:")
for t in test_texts:
    s, c = predict_sentiment(t, model, tfidf, bpe_tokenizer)
    print(f"   '{t}'\n   → {s}  (confidence: {c})\n")

# -----------------------------------
# Save All Artifacts
# -----------------------------------
print("💾 Saving model artifacts...")
joblib.dump(model,         "sentiment_model.pkl")
joblib.dump(tfidf,         "tfidf_vectorizer.pkl")
bpe_tokenizer.save(        "bpe_tokenizer.json")
print("✅ Saved: sentiment_model.pkl, tfidf_vectorizer.pkl, bpe_tokenizer.json")