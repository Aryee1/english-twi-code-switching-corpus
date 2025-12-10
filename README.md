# English–Twi Code-Switching Corpus and Language ID Baselines

This repository contains a small hand-annotated corpus of English–Twi code-switching and a set of simple language identification (LID) baselines.

The focus is Ghanaian conversational speech where Twi is usually the matrix language and English appears as embedded items. The project is aimed at low-resource NLP and code-switching, not at building a large production system.

Current goals:

1. Build a small conversational corpus with token-level language labels (English, Twi, other).
2. Train and compare:
   - a token-only LID baseline, and  
   - a simple context-aware model that uses within-utterance context and previous utterances in the conversation.

The current seed corpus is deliberately small (tens of utterances, a few hundred tokens). It is meant as a proof of concept and a starting point for future expansion.

---

## Repository structure

```text
english-twi-code-switching-corpus/
  data/
    raw/
      annotations.csv          # manually annotated utterances
    processed/
      lid_dataset.jsonl        # ML-ready dataset built from annotations.csv
    annotation_schema.md       # detailed description of the annotation scheme
  src/
    build_dataset.py           # CSV -> JSONL conversion with consistency checks
    train_lid_baseline.py      # token-only LID baselines
    train_lid_context_model.py # context-aware LID model
  models/                      # saved models (created after training)
  README.md
  requirements.txt
````

---

## Data and annotation

### Corpus

* Conversational data with English–Twi code-switching.
* Each row in `annotations.csv` is **one utterance**.
* The seed version currently has:

  * 44 utterances
  * 442 tokens
    (numbers will change if you add more data)

### Columns in `annotations.csv`

Each utterance has the following fields:

* `conv_id` – conversation ID (e.g. `conv001`, `conv028`)
* `utt_id` – utterance index within the conversation (starting at 1)
* `speaker` – anonymised speaker label (`A`, `B`, `C`, …)
* `text` – original utterance text (after anonymisation)
* `tokens` – whitespace-separated tokens
* `lang_tags` – language label per token
* `etymon_tags` – lexical origin per token (English/Twi/other)
* `token_role_tags` – token role (`cont`, `func`, `affix`, etc.)
* `utt_matrix_lang` – matrix language for the utterance (`tw`, `en`, `oth`)
* `cs_type` – code-switching type (`none`, `inter`, `intra`, `mixed`)

See `data/annotation_schema.md` for the full schema, examples, and guidelines.

### Language labels

Token-level `lang_tags`:

* `en` – English
* `tw` – Twi
* `oth` – other (names, numbers, emojis, URLs, Pidgin, etc.)

The annotation is influenced by work on Twi–English code-switching (Myers-Scotton’s MLF model, mixed verb phrases, copula constructions, English adjectives inside Twi frames, etc.). Twi is treated as the matrix language in most examples.

---

## From annotations to ML dataset

The ML-ready dataset lives in `data/processed/lid_dataset.jsonl`.

To (re)build it from the CSV:

```bash
python src/build_dataset.py
```

What this script does:

* Reads `data/raw/annotations.csv`.
* Splits `tokens`, `lang_tags`, `etymon_tags`, `token_role_tags` on whitespace.
* Checks that all token-level fields have the same length per utterance.
* Writes one JSON object per utterance to `data/processed/lid_dataset.jsonl`.

Each JSON object has (at least):

```json
{
  "conv_id": "conv001",
  "utt_id": 1,
  "speaker": "A",
  "text": "Today mekɔ school",
  "tokens": ["Today", "me", "kɔ", "school"],
  "lang_tags": ["en", "tw", "tw", "en"],
  "etymon_tags": ["en", "tw", "tw", "en"],
  "token_role_tags": ["cont", "func", "cont", "cont"],
  "utt_matrix_lang": "tw",
  "cs_type": "intra"
}
```

---

## Baselines

All models are simple and meant as baselines, not as final systems.

### 1. Token-only LID baseline

Script: `src/train_lid_baseline.py`

What it does:

* Loads `lid_dataset.jsonl`.
* Flattens to a list of (token, language tag) pairs.
* Splits train/test by **conversation ID** (so no conversation appears in both sets).
* Trains two baselines on the **training** tokens only:

  1. **Majority baseline**

     * Always predicts the most frequent tag in the training data.
  2. **Lexicon baseline**

     * For each token type (lowercased), stores the most frequent tag seen in training.
     * At test time:

       * if the token has been seen before, predict that tag;
       * otherwise, fall back to the majority tag.

Outputs:

* Overall accuracy on the test set.
* Per-class precision/recall/f1 for `en`, `tw`, `oth`.

On the current seed corpus, the lexicon baseline is only slightly better than the majority baseline and systematically over-predicts Twi, missing many English tokens.

Run:

```bash
python src/train_lid_baseline.py
```

---

### 2. Context-aware LID model

Script: `src/train_lid_context_model.py`

This is still a per-token classifier, but it uses:

* **Token-level features**

  * lowercase token
  * character suffixes
  * simple shape features (has digit, has diacritics, casing)
* **Within-utterance context**

  * previous token (`prev_tok`)
  * next token (`next_tok`)
  * position in utterance (`start`, `mid`, `end`)
  * token role (`token_role_tags` if available)
* **Utterance-level context**

  * matrix language (`utt_matrix_lang`)
  * speaker ID
  * utterance position bucket in the conversation (`1`, `2-3`, `4plus`)
* **Previous-utterance context**

  * whether a previous utterance exists in the same conversation
  * previous utterance matrix language
  * whether the speaker is the same as in the previous utterance

Model:

* Feature extractor → `DictVectorizer`
* Classifier → `LogisticRegression` (multiclass) from scikit-learn
* Train/test split by conversation ID, same as the baselines.

Run:

```bash
python src/train_lid_context_model.py
```

On the current data, the context-aware model clearly outperforms the token-only baselines, especially on English tokens: it recovers far more English tokens correctly while maintaining high accuracy on Twi.

---

## Setup

Tested with:

* Python 3.x
* `scikit-learn`
* `numpy`
* `scipy`
* `joblib`
* `tqdm`

Install:

```bash
pip install -r requirements.txt
```

---

## What this project is (and is not)

This repo is:

* A small, hand-annotated English–Twi code-switching corpus.
* A concrete example of low-resource LID experiments in a Ghanaian setting.
* A pair of simple baselines (token-only vs context-aware) showing how far you can get with modest data and simple models.

It is not:

* A large-scale or production-ready dataset.
* A state-of-the-art system for code-switching ASR or MT.

The idea is to provide a clean, inspectable example that can be extended with more data, richer features, or more advanced models in future work.
