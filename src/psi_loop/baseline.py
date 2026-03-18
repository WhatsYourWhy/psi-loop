"""Similarity-only baseline for comparison with Psi0."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from psi_loop.embedders import Embedder
from psi_loop.models import Candidate, ScoredCandidate, SelectionResult
from psi_loop.pipeline import fit_to_budget, rank_candidates
from psi_loop.scoring import goal_similarity


def baseline_score(
    candidate_text: str,
    goal: str,
    current_context: Iterable[str],
    embedder: Embedder | None = None,
) -> tuple[float, float, float]:
    """Similarity-only scorer compatible with the shared pipeline."""

    del current_context
    score = goal_similarity(candidate_text, goal, embedder=embedder)
    return score, score, 0.0


def rank_candidates_baseline(
    candidates: Sequence[Candidate],
    goal: str,
    embedder: Embedder | None = None,
) -> list[ScoredCandidate]:
    """Rank candidates by plain goal similarity."""

    ranked = rank_candidates(
        candidates,
        goal=goal,
        current_context=[],
        scorer=baseline_score,
        embedder=embedder,
    )
    return sorted(ranked, key=lambda item: (-item.score, item.candidate.id))


def select_context_baseline(
    candidates: Sequence[Candidate],
    goal: str,
    max_tokens: int,
    embedder: Embedder | None = None,
) -> SelectionResult:
    """Rank and select with a similarity-only baseline.

    Dense cosine similarity can be negative for anti-goal candidates. Keep those
    items in the ranked list for inspection, but do not allow them into the
    selected context window.
    """

    ranked = rank_candidates_baseline(candidates, goal, embedder=embedder)
    selected = fit_to_budget([item for item in ranked if item.score > 0.0], max_tokens)
    return SelectionResult(ranked=ranked, selected=selected, max_tokens=max_tokens)
