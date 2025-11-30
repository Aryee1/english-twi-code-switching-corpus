import json
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/lid_dataset.jsonl")
MODEL_DIR = Path("models")
SEP_TOKEN = "[SEP]"  # separator between token and previous utterance


def load_examples_with_context(path: Path) -> Tuple[List[str], List[str]]:
    """
    For each token, create a feature string that optionally includes the previous utterance
    in the same conversation: 'token [SEP] previous_utterance_text'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    # First read all records
    convs: Dict[str, List[dict]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            conv_id = ex["conv_id"]
            convs.setdefault(conv_id, []).append(ex)

    # Sort utterances within each conversation by utt_id
    for conv_id in convs:
        convs[conv_id] = sorted(convs[conv_id], key=lambda r: r["utt_id"])

    texts: List[str] = []
    labels: List[str] = []

    # Now build token-level examples with context
    for conv_id, utt_list in convs.items():
        prev_text = ""  # reset at start of each conversation

        for ex in utt_list:
            tokens = ex["tokens"]
            tags = ex["lang_tags"]

            for token, tag in zip(tokens, tags):
                if prev_text:
                    feature_text = f"{token} {SEP_TOKEN} {prev_text}"
                else:
                    feature_text = token  # no previous utterance yet

                texts.append(feature_text)
                labels.append(tag)

            # update previous utterance text for next turn
            prev_text = ex["text"]

    return texts, labels


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    texts, labels = load_examples_with_context(DATA_PATH)
    print(f"Loaded {len(texts)} token examples with context")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        lowercase=True,
    )

    print("Fitting context vectorizer...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training context-aware logistic regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    print("\n=== Context-aware model performance (token + previous utterance) ===")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, y_pred, labels=sorted(set(labels))))
    print("Labels order:", sorted(set(labels)))

    # Save models
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer_context.joblib")
    joblib.dump(clf, MODEL_DIR / "lid_context.joblib")
    print(f"\nSaved context-aware model and vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
