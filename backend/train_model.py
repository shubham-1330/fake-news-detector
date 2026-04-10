import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Download stopwords
nltk.download('stopwords')

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine title + text (IMPORTANT)
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

# Merge datasets
data = pd.concat([fake, true], axis=0)

# Keep needed columns
data = data[["content", "label"]]
data = data.rename(columns={"content": "text"})

# Load Indian dataset (if exists)
try:
    indian = pd.read_csv("indian_news.csv")
    data = pd.concat([data, indian], axis=0)
    print("Indian dataset added ✅")
except:
    print("No Indian dataset found (optional)")

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

# Show label distribution
print(data["label"].value_counts())

# TF-IDF (improved)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), max_df=0.7)
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (balanced)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved!")