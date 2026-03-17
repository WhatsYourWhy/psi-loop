"""Similarity-only baseline for comparison with Psi0."""

from __future__ import annotations

from collections.abc import Sequence

from psi_loop.models import Candidate, ScoredCandidate, SelectionResult
from psi_loop.pipeline import fit_to_budget
from psi_loop.scoring import goal_similarity, tokenize


def rank_candidates_baseline(candidates: Sequence[Candidate], goal: str) -> list[ScoredCandidate]:
    """Rank candidates by plain goal similarity."""

    ranked: list[ScoredCandidate] = []
    for candidate in candidates:
        ranked.append(
            ScoredCandidate(
                candidate=candidate,
                score=goal_similarity(candidate.text, goal),
                value=goal_similarity(candidate.text, goal),
                surprise=0.0,
                token_count=len(tokenize(candidate.text)),
            )
        )

    return sorted(ranked, key=lambda item: (-item.score, item.candidate.id))


def select_context_baseline(
    candidates: Sequence[Candidate],
    goal: str,
    max_tokens: int,
) -> SelectionResult:
    """Rank and select with a similarity-only baseline."""

    ranked = rank_candidates_baseline(candidates, goal)
    selected = fit_to_budget(ranked, max_tokens)
    return SelectionResult(ranked=ranked, selected=selected, max_tokens=max_tokens)
