"""Deterministic scoring helpers for the Psi0 prototype."""

from __future__ import annotations

from collections.abc import Iterable

from psi_loop.embedders import BowEmbedder, Embedder, centroid, cosine_similarity_vectors
from psi_loop.text import tokenize

GENERIC_GOAL_TERMS = {
    "api",
    "best",
    "brief",
    "client",
    "design",
    "discussion",
    "flow",
    "improve",
    "logic",
    "memo",
    "new",
    "note",
    "plan",
    "prepare",
    "python",
    "query",
    "review",
    "roadmap",
    "safer",
    "select",
    "summary",
    "system",
    "task",
}

ACTION_MECHANISM_TERMS = {
    "backoff",
    "contract",
    "dead",
    "event",
    "expand",
    "feedback",
    "guardrail",
    "idempotency",
    "invalidate",
    "invalidation",
    "jitter",
    "letter",
    "migration",
    "queue",
    "retention",
    "retry",
    "rollout",
}

LOW_WEIGHT = 0.25
DEFAULT_WEIGHT = 1.0
HIGH_WEIGHT = 4.0


def _resolve_embedder(embedder: Embedder | None) -> Embedder:
    return embedder if embedder is not None else BowEmbedder()


def cosine_similarity(left_text: str, right_text: str, embedder: Embedder | None = None) -> float:
    """Convenience wrapper over the selected embedder."""

    scorer = _resolve_embedder(embedder)
    return cosine_similarity_vectors(scorer.embed(left_text), scorer.embed(right_text))


def goal_term_weight(token: str) -> float:
    """Weight goal terms by procedural relevance rather than flat overlap."""

    if token in ACTION_MECHANISM_TERMS:
        return HIGH_WEIGHT
    if token in GENERIC_GOAL_TERMS:
        return LOW_WEIGHT
    return DEFAULT_WEIGHT


def keyword_overlap(candidate_text: str, goal: str) -> float:
    """Value proxy: normalized weighted overlap over unique goal tokens."""

    goal_tokens = set(tokenize(goal))
    if not goal_tokens:
        return 0.0

    candidate_tokens = set(tokenize(candidate_text))
    total_goal_weight = sum(goal_term_weight(token) for token in goal_tokens)
    if total_goal_weight == 0.0:
        return 0.0

    matched_goal_weight = sum(
        goal_term_weight(token)
        for token in goal_tokens
        if token in candidate_tokens
    )
    return matched_goal_weight / total_goal_weight


def surprise_score(
    candidate_text: str,
    current_context: Iterable[str],
    embedder: Embedder | None = None,
) -> float:
    """Surprise proxy: distance from the current context centroid."""

    context_items = [part for part in current_context if part.strip()]
    if not context_items:
        return 1.0

    scorer = _resolve_embedder(embedder)
    candidate_vector = scorer.embed(candidate_text)
    context_vectors = [scorer.embed(part) for part in context_items]
    context_centroid = centroid(context_vectors)
    similarity = cosine_similarity_vectors(candidate_vector, context_centroid)
    return max(0.0, min(1.0, 1.0 - similarity))


def goal_similarity(candidate_text: str, goal: str, embedder: Embedder | None = None) -> float:
    """Baseline similarity-only score against the goal."""

    return cosine_similarity(candidate_text, goal, embedder=embedder)


def psi_0(
    candidate_text: str,
    goal: str,
    current_context: Iterable[str],
    embedder: Embedder | None = None,
) -> tuple[float, float, float]:
    """Return overall Psi0 score plus its value and surprise components."""

    value = keyword_overlap(candidate_text, goal)
    surprise = surprise_score(candidate_text, current_context, embedder=embedder)
    return value * surprise, value, surprise
