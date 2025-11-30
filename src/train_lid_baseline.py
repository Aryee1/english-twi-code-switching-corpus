import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/lid_dataset.jsonl")
MODEL_DIR = Path("models")


def load_token_level_examples(path: Path) -> Tuple[List[str], List[str]]:
    """Load one training example per token."""
    texts: List[str] = []
    labels: List[str] = []

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            tokens = example["tokens"]
            tags = example["lang_tags"]

            for token, tag in zip(tokens, tags):
                texts.append(token)
                labels.append(tag)

    return texts, labels


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    texts, labels = load_token_level_examples(DATA_PATH)
    print(f"Loaded {len(texts)} tokens")

    # If the dataset is small, this split will be tiny, but that is okay for now.
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    # Character-level TF-IDF features (good for different spelling patterns)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        lowercase=True,
    )

    print("Fitting vectorizer...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training logistic regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    print("\n=== Sentence-only baseline performance ===")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, y_pred, labels=sorted(set(labels))))
    print("Labels order:", sorted(set(labels)))

    # Save models
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer_baseline.joblib")
    joblib.dump(clf, MODEL_DIR / "lid_baseline.joblib")
    print(f"\nSaved baseline model and vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
