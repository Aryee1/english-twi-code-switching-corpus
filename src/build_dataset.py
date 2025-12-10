#!/usr/bin/env python3
"""
Build LID dataset from annotations.csv.

- Input:  data/raw/annotations.csv
- Output: data/processed/lid_dataset.jsonl
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "annotations.csv"
OUT_JSONL = ROOT / "data" / "processed" / "lid_dataset.jsonl"


def split_field(s: str):
    """Split a whitespace-separated field into a list, handling empty values."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return s.split()


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Could not find {RAW_CSV}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with RAW_CSV.open("r", encoding="utf-8") as f_in, \
         OUT_JSONL.open("w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        required_cols = [
            "conv_id", "utt_id", "speaker", "text",
            "tokens", "lang_tags"
        ]
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column in CSV: {col}")

        n_rows = 0
        for row in reader:
            conv_id = row["conv_id"]
            utt_id = int(row["utt_id"])
            speaker = row["speaker"]
            text = row["text"]

            tokens = split_field(row.get("tokens", ""))
            lang_tags = split_field(row.get("lang_tags", ""))

            etymon_tags = split_field(row.get("etymon_tags", ""))
            token_role_tags = split_field(row.get("token_role_tags", ""))

            utt_matrix_lang = row.get("utt_matrix_lang", "").strip() or None
            cs_type = row.get("cs_type", "").strip() or None

            # Basic consistency checks
            if len(tokens) != len(lang_tags):
                raise ValueError(
                    f"Length mismatch in {conv_id}, utt {utt_id}: "
                    f"{len(tokens)} tokens vs {len(lang_tags)} lang_tags"
                )

            if etymon_tags and len(etymon_tags) != len(tokens):
                raise ValueError(
                    f"Length mismatch in {conv_id}, utt {utt_id}: "
                    f"{len(tokens)} tokens vs {len(etymon_tags)} etymon_tags"
                )

            if token_role_tags and len(token_role_tags) != len(tokens):
                raise ValueError(
                    f"Length mismatch in {conv_id}, utt {utt_id}: "
                    f"{len(tokens)} tokens vs {len(token_role_tags)} token_role_tags"
                )

            # Build JSON object
            example = {
                "conv_id": conv_id,
                "utt_id": utt_id,
                "speaker": speaker,
                "text": text,
                "tokens": tokens,
                "lang_tags": lang_tags,
                "etymon_tags": etymon_tags if etymon_tags else None,
                "token_role_tags": token_role_tags if token_role_tags else None,
                "utt_matrix_lang": utt_matrix_lang,
                "cs_type": cs_type,
            }

            # Optional: drop None fields to keep JSON clean
            example = {k: v for k, v in example.items() if v is not None}

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_rows += 1

    print(f"Wrote {n_rows} utterances to {OUT_JSONL}")


if __name__ == "__main__":
    main()
