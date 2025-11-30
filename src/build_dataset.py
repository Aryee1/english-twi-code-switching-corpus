import csv
import json
from pathlib import Path

# Paths
RAW_PATH = Path("data/raw/annotations.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "lid_dataset.jsonl"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Could not find input file at {RAW_PATH}")

    num_rows = 0
    num_written = 0

    with RAW_PATH.open(newline="", encoding="utf-8") as csvfile, \
         OUT_PATH.open("w", encoding="utf-8") as out:

        reader = csv.DictReader(csvfile)
        for row in reader:
            num_rows += 1

            # Split tokens and tags on spaces
            tokens = str(row["tokens"]).split()
            tags = str(row["lang_tags"]).split()

            if len(tokens) != len(tags):
                print(
                    f"[WARN] Skipping row with length mismatch: "
                    f"conv_id={row.get('conv_id')} utt_id={row.get('utt_id')} "
                    f"(tokens={len(tokens)}, tags={len(tags)})"
                )
                continue

            record = {
                "conv_id": str(row["conv_id"]),
                "utt_id": int(row["utt_id"]),
                "speaker": str(row["speaker"]),
                "text": str(row["text"]),
                "tokens": tokens,
                "lang_tags": tags,
            }

            json_line = json.dumps(record, ensure_ascii=False)
            out.write(json_line + "\n")
            num_written += 1

    print(f"Read {num_rows} CSV rows")
    print(f"Wrote {num_written} JSONL records to {OUT_PATH}")


if __name__ == "__main__":
    main()
