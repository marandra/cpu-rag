"""
BM25 tokenizers for Spanish medical text.

Available tokenizers:
    "whitespace"  — basic text.lower().split() (baseline, no NLP)
    "spacy"       — spaCy es_core_news_sm: lemmatization + stopword removal + accent normalization

All tokenizers accept a string and return a list of tokens.
"""

from __future__ import annotations

import unicodedata
from typing import Callable


def strip_accents(text: str) -> str:
    """Remove diacritical marks (á→a, ñ→n, ü→u)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def whitespace_tokenizer(text: str) -> list[str]:
    """Baseline tokenizer: lowercase + split on whitespace."""
    return text.lower().split()


def whitespace_accent_tokenizer(text: str) -> list[str]:
    """Whitespace tokenizer with accent normalization."""
    return strip_accents(text.lower()).split()


_spacy_nlp = None


def _get_spacy():
    """Lazy-load spaCy model (singleton)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        _spacy_nlp = spacy.load(
            "es_core_news_sm",
            disable=["ner", "parser"],  # we only need tokenizer + lemmatizer
        )
    return _spacy_nlp


def spacy_tokenizer(text: str) -> list[str]:
    """spaCy Spanish tokenizer: lemmatize + remove stopwords + strip accents.

    Returns lowercased, accent-normalized lemmas. Filters out:
    - Stopwords (de, la, en, que, ...)
    - Punctuation and whitespace
    - Single-character tokens
    """
    nlp = _get_spacy()
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = strip_accents(token.lemma_)
        if len(lemma) > 1:
            tokens.append(lemma)
    return tokens


# --- Registry ---

TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "whitespace": whitespace_tokenizer,
    "whitespace+accent": whitespace_accent_tokenizer,
    "spacy": spacy_tokenizer,
}


def get_tokenizer(name: str = "spacy") -> Callable[[str], list[str]]:
    """Get a tokenizer by name."""
    if name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer '{name}'. Choose from: {list(TOKENIZERS.keys())}")
    return TOKENIZERS[name]
