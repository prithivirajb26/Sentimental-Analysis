from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
from tokenizers import Tokenizer
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------
# Load All Model Artifacts
# ----------------------------------
model         = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
tfidf         = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
bpe_tokenizer = Tokenizer.from_file(os.path.join(BASE_DIR, "bpe_tokenizer.json"))

BPE_VOCAB_SIZE = 3000

print("✅ Model, TF-IDF, and BPE tokenizer loaded successfully.")

# ----------------------------------
# Negation Words
# ----------------------------------
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor",
    "hardly", "barely", "scarcely", "without",
    "nobody", "nothing", "nowhere", "none",
    "isnt", "wasnt", "doesnt", "didnt", "wont",
    "cant", "couldnt", "shouldnt", "wouldnt",
    "havent", "hasnt", "hadnt", "arent", "werent"
}

# Filler words — skip these when looking for the word to negate
# Example: "not a bad boy" → skip "a", flip "bad" → "NOT_bad"
FILLER_WORDS = {
    "a", "an", "the", "very", "really", "so",
    "such", "quite", "rather", "too", "just",
    "that", "this", "my", "your", "his", "her",
    "our", "their", "its", "any", "some", "much",
    "more", "most", "even", "only", "ever"
}

# ----------------------------------
# Negation Handler (with filler skip)
# ----------------------------------
def handle_negation(text):
    """
    Flips the meaning of the first meaningful word after a negation word.
    Filler words (a, an, the, very, really...) are skipped so the correct
    word gets flipped.

    Examples:
      "not bad"           → "not NOT_bad"
      "not a bad boy"     → "not a NOT_bad boy"     ✅ skips "a"
      "never a good idea" → "never a NOT_good idea" ✅ skips "a"
      "not very happy"    → "not very NOT_happy"    ✅ skips "very"
      "not really great"  → "not really NOT_great"  ✅ skips "really"
    """
    words = text.split()
    result = []
    negate = False

    for word in words:
        clean_word = re.sub(r"[^a-z]", "", word)

        if clean_word in NEGATION_WORDS:
            # Found a negation word — activate negate mode
            negate = True
            result.append(word)

        elif negate and clean_word in FILLER_WORDS:
            # This is a filler word — skip it but keep negate ON
            # So we keep looking for the real word to flip
            result.append(word)

        elif negate and clean_word:
            # This is the real meaningful word — flip it
            result.append("NOT_" + clean_word)
            negate = False  # reset after flipping

        else:
            result.append(word)

    return " ".join(result)

# ----------------------------------
# Text Cleaning
# ----------------------------------
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

# ----------------------------------
# BPE → Bag-of-Tokens Matrix
# ----------------------------------
def bpe_to_bow(texts, tokenizer, vocab_size=3000):
    rows, cols, data = [], [], []
    for i, text in enumerate(texts):
        encoding = tokenizer.encode(str(text))
        for tid in encoding.ids:
            if tid < vocab_size:
                rows.append(i)
                cols.append(tid)
                data.append(1)
    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(texts), vocab_size)
    )

# ----------------------------------
# Confidence Adjuster
# ----------------------------------
def adjust_confidence(text_original, confidence):
    """
    Softens confidence when negation is detected since negation
    sentences are genuinely harder for the model to classify.
    """
    lower = text_original.lower()
    negation_found = any(neg in lower.split() for neg in NEGATION_WORDS)
    has_contraction = bool(re.search(r"n't|cannot|won't", lower))

    if negation_found or has_contraction:
        confidence = round(min(confidence, 0.80), 4)

    return confidence

# ----------------------------------
# Combined Prediction
# ----------------------------------
def predict(text):
    cleaned   = clean_text(text)
    tfidf_vec = tfidf.transform([cleaned])
    bpe_vec   = bpe_to_bow([cleaned], bpe_tokenizer, BPE_VOCAB_SIZE)
    combined  = hstack([tfidf_vec, bpe_vec])
    pred      = model.predict(combined)[0]
    proba     = model.predict_proba(combined)[0].tolist()
    sentiment = "positive" if pred in [1, 4] else "negative"
    confidence = round(max(proba), 4)
    confidence = adjust_confidence(text, confidence)
    return sentiment, confidence

# ----------------------------------
# Routes
# ----------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sentiment Analysis API (BPE + TF-IDF + Negation Handling) is running",
        "model":   "Logistic Regression with BPE + TF-IDF + Smart Negation",
        "endpoints": {
            "POST /predict": "Analyze sentiment of a text",
            "GET /health":   "Health check"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "BPE + TF-IDF + Smart Negation Handling"})

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    sentiment, confidence = predict(text)

    return jsonify({
        "text":       text,
        "sentiment":  sentiment,
        "confidence": confidence,
        "method":     "BPE + TF-IDF + Smart Negation Handling"
    })

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)