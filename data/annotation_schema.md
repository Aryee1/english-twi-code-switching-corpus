# Annotation schema for the English–Twi code-switching corpus

This document describes how utterances are annotated in the corpus.

Each utterance in `data/raw/annotations.csv` has the following fields:

- `conv_id`  
  Identifier for the conversation. Example: `conv1`, `conv2`.

- `utt_id`  
  Integer index of the utterance within the conversation. The first turn is `1`.

- `speaker`  
  Speaker label such as `A`, `B`, `C`. These labels are anonymized and do not correspond to real names.

- `text`  
  The original utterance text after anonymization.

- `tokens`  
  Word-level tokens, separated by spaces in the CSV. The order matches the original utterance.

- `lang_tags`  
  Language labels for each token, separated by spaces. The number of labels must match the number of tokens.

## Language labels

The corpus focuses on English–Twi code-switching. The following labels are used:

- `en` – English tokens  
- `tw` – Twi tokens  
- `oth` – Other items such as:
  - Names  
  - Emojis  
  - URLs  
  - Numbers  
  - Punctuation (when tokenized)  
  - Any token that is not clearly English or Twi

Tokens that belong to Ghanaian Pidgin are generally **not** the focus of this corpus. If they appear, they can be labelled as `oth`, but the main analysis targets English–Twi alternation.

## Examples

### Example 1

Text:  
`Today mekor school`

Tokens:  
`Today mekor school`

Language tags:  
`en tw en`

Explanation:
- `Today`, `school` are English  
- `mekor` is Twi in this context

---

### Example 2

Text:  
`Okay but fa book no`

Tokens:  
`Okay but fa book no`

Language tags:  
`en en tw en en`

Explanation:
- `Okay`, `but`, `book`, `no` are English  
- `fa` is Twi  

---

### Example 3

Text:  
`Me pɛ sɛ I rest small`

Tokens:  
`Me pɛ sɛ I rest small`

Language tags:  
`tw tw tw en en en`

Explanation:
- `Me pɛ sɛ` is Twi  
- `I rest small` is English in this context

---

## General guidelines

1. Tokenization is word-based. Do not split inside a normal word.
2. Make sure `tokens` and `lang_tags` have the same number of items for every utterance.
3. If you are unsure whether a token is English or Twi, decide on a label and use it consistently. If it does not clearly belong to either, use `oth`.
4. Personal names, phone numbers, or other identifying details should be removed or anonymized. If they remain as tokens, label them as `oth`.
5. Conversations are defined by `conv_id`, and `utt_id` preserves the order of turns within each conversation. This structure is used by the context-aware model to access previous utterances.
