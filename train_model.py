"""Train and compare text-classification models for fake-news detection."""
from pathlib import Path
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

DATA_PATH = Path("News_dataset.csv")
MODEL_PATH = Path("fake_news_pipeline.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("News_dataset.csv is missing. Place it in the project root.")
    df = pd.read_csv(DATA_PATH)
    if not {"title", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'title' and 'label' columns.")
    df = df[["title", "label"]].dropna().drop_duplicates(subset=["title"])
    df["title"] = df["title"].astype(str).map(clean_text)
    df = df[df["title"].str.len() > 0]
    X_train, X_test, y_train, y_test = train_test_split(
        df["title"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }
    best_name, best_pipeline, best_f1 = None, None, -1
    for name, classifier in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100000)),
            ("classifier", classifier),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = f1_score(y_test, pred, average="weighted")
        print(f"\n=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_test, pred), 4))
        print("Precision:", round(precision_score(y_test, pred, average="weighted", zero_division=0), 4))
        print("Recall:", round(recall_score(y_test, pred, average="weighted", zero_division=0), 4))
        print("Weighted F1:", round(score, 4))
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        if score > best_f1:
            best_name, best_pipeline, best_f1 = name, pipe, score
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model: {best_name} (weighted F1={best_f1:.4f})")
    print(f"Saved pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
