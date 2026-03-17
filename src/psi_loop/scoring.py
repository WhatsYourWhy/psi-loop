"""Deterministic scoring helpers for the Psi0 prototype."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

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


def cosine_similarity_from_counts(left: Counter[str], right: Counter[str]) -> float:
    """Cosine similarity for sparse bag-of-words vectors."""

    if not left or not right:
        return 0.0

    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cosine_similarity(left_text: str, right_text: str) -> float:
    """Convenience wrapper over token counts."""

    return cosine_similarity_from_counts(token_counts(left_text), token_counts(right_text))


def keyword_overlap(candidate_text: str, goal: str) -> float:
    """Value proxy: fraction of goal keywords present in the candidate."""

    goal_tokens = set(tokenize(goal))
    if not goal_tokens:
        return 0.0

    candidate_tokens = set(tokenize(candidate_text))
    overlap = goal_tokens & candidate_tokens
    return len(overlap) / len(goal_tokens)


def surprise_score(candidate_text: str, current_context: Iterable[str]) -> float:
    """Surprise proxy: distance from the current context centroid."""

    context_text = " ".join(part for part in current_context if part.strip())
    if not context_text.strip():
        return 1.0

    similarity = cosine_similarity(candidate_text, context_text)
    return max(0.0, 1.0 - similarity)


def goal_similarity(candidate_text: str, goal: str) -> float:
    """Baseline similarity-only score against the goal."""

    return cosine_similarity(candidate_text, goal)


def psi_0(candidate_text: str, goal: str, current_context: Iterable[str]) -> tuple[float, float, float]:
    """Return overall Psi0 score plus its value and surprise components."""

    value = keyword_overlap(candidate_text, goal)
    surprise = surprise_score(candidate_text, current_context)
    return value * surprise, value, surprise
