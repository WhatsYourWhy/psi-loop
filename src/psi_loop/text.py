"""Shared text normalization helpers."""

from __future__ import annotations

import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "with",
}


def _normalize_token(token: str) -> str:
    """Apply a tiny amount of stemming to keep fixtures readable."""

    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Normalize text into a deterministic token list."""

    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    return [
        _normalize_token(token)
        for token in raw_tokens
        if token not in STOPWORDS
    ]


def token_counts(text: str) -> Counter[str]:
    """Build a sparse token frequency vector."""

    return Counter(tokenize(text))
