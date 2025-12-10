#!/usr/bin/env python3
"""
Context-aware LID model for the English–Twi code-switching corpus.

- Uses token form + within-utterance context + previous-utterance context.
- Model: logistic regression over feature dictionaries.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "lid_dataset.jsonl"


def load_dataset(path: Path) -> List[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def group_by_conv(examples: List[dict]) -> Dict[str, List[dict]]:
    """Group utterances by conversation and sort by utt_id."""
    convs: Dict[str, List[dict]] = defaultdict(list)
    for ex in examples:
        convs[ex["conv_id"]].append(ex)
    for cid in convs:
        convs[cid].sort(key=lambda e: e["utt_id"])
    return convs


def token_shape_features(tok: str) -> Dict[str, Any]:
    """Simple orthographic features."""
    feats: Dict[str, Any] = {}
    lower = tok.lower()

    feats["tok"] = lower
    feats["suffix2"] = lower[-2:] if len(lower) >= 2 else lower
    feats["suffix3"] = lower[-3:] if len(lower) >= 3 else lower

    feats["is_ascii"] = all(ord(ch) < 128 for ch in tok)
    feats["has_digit"] = any(ch.isdigit() for ch in tok)
    feats["is_upper"] = tok.isupper()

    # crude diacritic detection: non-ASCII letters
    feats["has_diacritic"] = any(ord(ch) > 127 and ch.isalpha() for ch in tok)

    return feats


def utt_position_bucket(utt_id: int) -> str:
    if utt_id == 1:
        return "1"
    elif utt_id <= 3:
        return "2-3"
    else:
        return "4plus"


def extract_features_from_conv(conv_utts: List[dict]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Given a list of utterances in one conversation (sorted by utt_id),
    return feature dicts and gold labels for all tokens.
    """
    X_feats: List[Dict[str, Any]] = []
    y_labels: List[str] = []

    n_utts = len(conv_utts)

    for ui, utt in enumerate(conv_utts):
        tokens = utt["tokens"]
        tags = utt["lang_tags"]
        utt_id = utt["utt_id"]
        speaker = utt.get("speaker", "UNK")
        utt_matrix = utt.get("utt_matrix_lang", "unk") or "unk"
        token_roles = utt.get("token_role_tags", None)

        # previous utterance context
        if ui > 0:
            prev_utt = conv_utts[ui - 1]
            prev_matrix = prev_utt.get("utt_matrix_lang", "none") or "none"
            prev_speaker = prev_utt.get("speaker", "UNK")
            prev_exists = "yes"
            same_speaker = "yes" if prev_speaker == speaker else "no"
        else:
            prev_utt = None
            prev_matrix = "none"
            prev_exists = "no"
            same_speaker = "no"

        # token-level
        for ti, (tok, tag) in enumerate(zip(tokens, tags)):
            feats: Dict[str, Any] = {}

            # shape features
            feats.update(token_shape_features(tok))

            # within-utterance context
            if ti == 0:
                prev_tok = "<BOS>"
                position = "start"
            else:
                prev_tok = tokens[ti - 1].lower()
                position = "mid"

            if ti == len(tokens) - 1:
                next_tok = "<EOS>"
                if position == "mid":
                    position = "end"
            else:
                next_tok = tokens[ti + 1].lower()

            feats["prev_tok"] = prev_tok
            feats["next_tok"] = next_tok
            feats["position_in_utt"] = position

            # token role if available
            if token_roles:
                role = token_roles[ti]
                feats["token_role"] = role
            else:
                feats["token_role"] = "none"

            # utterance-level
            feats["utt_matrix_lang"] = utt_matrix
            feats["speaker"] = speaker
            feats["utt_id_bucket"] = utt_position_bucket(utt_id)

            # previous-utterance context
            feats["prev_utt_exists"] = prev_exists
            feats["prev_utt_matrix_lang"] = prev_matrix
            feats["same_speaker_as_prev"] = same_speaker

            X_feats.append(feats)
            y_labels.append(tag)

    return X_feats, y_labels


def build_dataset_for_context_model(examples: List[dict]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Build feature dicts and labels for all tokens,
    and also return a list of conversation IDs per token for splitting.
    """
    convs = group_by_conv(examples)
    X_all: List[Dict[str, Any]] = []
    y_all: List[str] = []
    conv_ids_per_token: List[str] = []

    for cid, utts in convs.items():
        X_feats, y_labels = extract_features_from_conv(utts)
        X_all.extend(X_feats)
        y_all.extend(y_labels)
        conv_ids_per_token.extend([cid] * len(y_labels))

    return X_all, y_all, conv_ids_per_token


def split_by_conversation(conv_ids_per_token: List[str], test_size: float = 0.2, random_state: int = 42):
    """
    Produce train/test indices so that all tokens from the same conversation
    go either into train or into test (no leakage across convs).
    """
    unique_convs = sorted(set(conv_ids_per_token))

    train_convs, test_convs = train_test_split(
        unique_convs, test_size=test_size, random_state=random_state
    )
    train_convs = set(train_convs)
    test_convs = set(test_convs)

    train_idx = []
    test_idx = []

    for i, cid in enumerate(conv_ids_per_token):
        if cid in train_convs:
            train_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, test_idx


def main():
    print(f"Loading dataset from {DATA} ...")
    examples = load_dataset(DATA)
    print(f"Loaded {len(examples)} utterances")

    X_dicts, y, conv_ids_per_token = build_dataset_for_context_model(examples)
    print(f"Total tokens: {len(y)}")

    # Conversation-level split
    train_idx, test_idx = split_by_conversation(conv_ids_per_token, test_size=0.2, random_state=42)

    # Slice into train/test
    X_train_dicts = [X_dicts[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test_dicts = [X_dicts[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"Train tokens: {len(y_train)}, Test tokens: {len(y_test)}")

    # Vectorise feature dicts
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(X_train_dicts)
    X_test = vec.transform(X_test_dicts)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nContext-aware LID accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
