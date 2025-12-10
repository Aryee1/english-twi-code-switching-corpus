#!/usr/bin/env python3
"""
Token-only LID baseline for English–Twi code-switching.

- Majority baseline: always predict the most frequent tag in TRAIN.
- Lexicon baseline: token -> most frequent tag in TRAIN (lowercased),
  fallback to majority tag for unseen tokens.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


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
    convs: Dict[str, List[dict]] = defaultdict(list)
    for ex in examples:
        convs[ex["conv_id"]].append(ex)
    for cid in convs:
        convs[cid].sort(key=lambda e: e["utt_id"])
    return convs


def flatten_tokens(examples: List[dict], conv_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
    tokens = []
    tags = []
    conv_ids_per_token = []
    for ex in examples:
        cid = ex["conv_id"]
        for tok, tag in zip(ex["tokens"], ex["lang_tags"]):
            tokens.append(tok)
            tags.append(tag)
            conv_ids_per_token.append(cid)
    return tokens, tags, conv_ids_per_token


def split_by_conversation(conv_ids_per_token: List[str], test_size: float = 0.2, random_state: int = 42):
    unique_convs = sorted(set(conv_ids_per_token))
    train_convs, test_convs = train_test_split(
        unique_convs, test_size=test_size, random_state=random_state
    )
    train_convs = set(train_convs)
    test_convs = set(test_convs)

    train_idx, test_idx = [], []
    for i, cid in enumerate(conv_ids_per_token):
        if cid in train_convs:
            train_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, test_idx


def main():
    examples = load_dataset(DATA)
    convs = group_by_conv(examples)
    # Flatten with conv_ids per token
    tokens = []
    tags = []
    conv_ids_per_token = []
    for cid, utts in convs.items():
        for ex in utts:
            for tok, tag in zip(ex["tokens"], ex["lang_tags"]):
                tokens.append(tok)
                tags.append(tag)
                conv_ids_per_token.append(cid)

    print(f"Total tokens: {len(tokens)}")

    train_idx, test_idx = split_by_conversation(conv_ids_per_token, test_size=0.2, random_state=42)

    tok_train = [tokens[i] for i in train_idx]
    y_train = [tags[i] for i in train_idx]
    tok_test = [tokens[i] for i in test_idx]
    y_test = [tags[i] for i in test_idx]

    print(f"Train tokens: {len(y_train)}, Test tokens: {len(y_test)}")

    # Majority baseline (trained on TRAIN only)
    tag_counts = Counter(y_train)
    majority_tag, _ = tag_counts.most_common(1)[0]

    y_pred_majority = [majority_tag] * len(y_test)
    acc_majority = accuracy_score(y_test, y_pred_majority)
    print(f"\nMajority tag (train): {majority_tag}")
    print(f"Majority baseline accuracy (test): {acc_majority:.3f}")

    # Lexicon baseline: token -> most frequent tag in TRAIN
    token_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    for tok, tag in zip(tok_train, y_train):
        token_tag_counts[tok.lower()][tag] += 1

    lexicon: Dict[str, str] = {}
    for tok, counter in token_tag_counts.items():
        lexicon[tok] = counter.most_common(1)[0][0]

    y_pred_lex = []
    for tok in tok_test:
        t = tok.lower()
        if t in lexicon:
            y_pred_lex.append(lexicon[t])
        else:
            y_pred_lex.append(majority_tag)

    acc_lex = accuracy_score(y_test, y_pred_lex)
    print(f"Lexicon baseline accuracy (test): {acc_lex:.3f}")

    print("\nLexicon baseline classification report (test):")
    print(classification_report(y_test, y_pred_lex, digits=3))


if __name__ == "__main__":
    main()
