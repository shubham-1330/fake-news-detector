from flask import Flask, request, jsonify
import joblib
import re
import nltk
import requests
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
# Download stopwords
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load stopwords
stop_words = set(stopwords.words("english"))

# 🔑 PUT YOUR API KEY HERE
API_KEY = os.environ.get("API Key")

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Home route
@app.route("/")
def home():
    return "Fake News Detection API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data.get("text", "")

    # Clean input
    cleaned = clean_text(news)

    # -----------------------------
    # 🧠 ML PREDICTION
    # -----------------------------
    vectorized = vectorizer.transform([cleaned])
    proba = model.predict_proba(vectorized)[0]

    print("ML PROB:", proba)

    # -----------------------------
    # 🔍 NEWS API FACT CHECK
    # -----------------------------
    similarity_score = 0
    articles_found = 0

    try:
        # Better query (IMPORTANT FIX)
        query = " ".join(news.split()[:5])

        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={API_KEY}"
        response = requests.get(url)
        articles = response.json().get("articles", [])

        articles_found = len(articles)
        print("ARTICLES FOUND:", articles_found)

        if articles:
            texts = [
                (a["title"] or "") + " " + (a["description"] or "")
                for a in articles[:5]
            ]

            texts.append(news)

            tfidf = vectorizer.transform(texts)
            sim = cosine_similarity(tfidf[-1], tfidf[:-1])

            similarity_score = float(sim.max())

    except Exception as e:
        print("API ERROR:", e)

    print("SIMILARITY:", similarity_score)

    # -----------------------------
    # 🎯 FINAL DECISION (IMPROVED)
    # -----------------------------
    if similarity_score > 0.3:
        result = "Real News (Verified) ✅"
    elif proba[1] > 0.6:
        result = "Likely Real News ✅"
    elif proba[1] > 0.4:
        result = "Uncertain ⚠️"
    else:
        result = "Fake News 🛑"

    return jsonify({
        "prediction": result,
        "ml_real_probability": float(proba[1]),
        "fake_probability": float(proba[0]),
        "similarity_score": similarity_score,
        "articles_found": articles_found
    })

if __name__ == "__main__":
    app.run(debug=True)
