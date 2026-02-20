# Detection of Contradictions in Legal Text Using Various NLP Techniques

This project tackles **contradiction detection** in legal documents via **Natural Language Inference (NLI)**

Given a premise (e.g. a clause or finding) and a hypothesis (e.g. a claim or conclusion), the system classifies the relation as **Entailment**, **Contradiction**, or **Not Mentioned**. We combine classical NLP (tokenization, lemmatization, TF‑IDF, logistic regression) with transformer-based models: a fine-tuned **Legal English RoBERTa** for sequence classification and a **sentence-transformers** setup for contradiction-oriented semantic similarity. All implemented in a single Jupyter pipeline (EDA → preprocessing → baselines → BERT NLI → sentence embeddings).

---

## Overview

- **Task:** 3-class NLI on legal documents (premise vs hypothesis).
- **Labels:** `0` = Entailment, `1` = Contradiction, `2` = Not Mentioned.
- **Pipeline:** EDA → text preprocessing (NLTK, TF-IDF) → baseline ML (e.g. logistic regression) → fine-tuned **Legal English RoBERTa** for sequence classification → **sentence-transformers** setup for contradiction-focused similarity.

---

## Project structure

```
.
├── main.ipynb          # Full pipeline: EDA, preprocessing, ML baselines, BERT NLI, sentence-transformers
├── requirements.txt   # Python dependencies
└── README.md
```

**Note:** Model checkpoints (`saved_nli_model/`, `sentence_transformer_contradiction_model/`, `bert_contradiction/`) and `data/` are not in the repo (see [Setup](#setup)).

---