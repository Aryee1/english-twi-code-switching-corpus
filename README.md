# English–Twi Code-Switching Corpus and Language ID Baselines

This project explores English–Twi code-switching in Ghanaian conversational data.

It has two main goals:

1. Build a small conversational corpus with token-level language annotations (English, Twi, other).
2. Train baseline language identification (LID) models and compare a token-only approach with a simple context-aware model that uses previous utterances in the conversation.

The project is motivated by the difficulty of applying English-trained NLP tools to multilingual Ghanaian data and the lack of resources for low-resource African languages.

---

## Repository structure

```text
english-twi-code-switching-corpus/
  data/
    raw/
      annotations.csv          # manually annotated data (not tracked if sensitive)
    processed/
      lid_dataset.jsonl        # ML-ready dataset built from annotations.csv
    annotation_schema.md       # description of the annotation scheme
  src/
    build_dataset.py           # CSV -> JSONL conversion
    train_lid_baseline.py      # token-only LID baseline
    train_lid_context_model.py # context-aware LID model (previous utterance)
  models/                      # saved models (created after training)
  README.md
  requirements.txt
