import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD DATA
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # words -> numbers

print("Total messages:", len(df))
print(df["label"].value_counts())  # how many ham vs spam

# 2. SPLIT: 80% for learning, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. TEXT -> NUMBERS (computers can't read words)
tfidf = TfidfVectorizer(lowercase=True, stop_words="english")
X_train_vec = tfidf.fit_transform(X_train)  # learn vocabulary from train only
X_test_vec = tfidf.transform(X_test)        # just convert test

# 4. TRAIN TWO MODELS
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

# 5. EVALUATE
for name, model in [("Naive Bayes", nb), ("Logistic Regression", lr)]:
    pred = model.predict(X_test_vec)
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# 6. SAVE THE BEST MODEL (using Naive Bayes here)
joblib.dump(nb, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("\nSaved model.pkl and tfidf.pkl")

# 7. TRY YOUR OWN MESSAGES
tests = [
    "Congratulations! You won a free iPhone. Click here to claim now",
    "Hey, are we meeting for lunch tomorrow?",
]
print("\nPredictions (1 = spam, 0 = ham):")
for t in tests:
    print(t, "->", nb.predict(tfidf.transform([t]))[0])