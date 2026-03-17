"""Budgeted selection pipeline for Psi0-ranked candidates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from psi_loop.models import Candidate, ScoredCandidate, SelectionResult
from psi_loop.scoring import psi_0, tokenize


def _token_count(text: str) -> int:
    return len(tokenize(text))


def rank_candidates(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: Iterable[str],
) -> list[ScoredCandidate]:
    """Rank candidates with Psi0 and preserve useful scoring detail."""

    ranked: list[ScoredCandidate] = []
    for candidate in candidates:
        score, value, surprise = psi_0(candidate.text, goal, current_context)
        ranked.append(
            ScoredCandidate(
                candidate=candidate,
                score=score,
                value=value,
                surprise=surprise,
                token_count=_token_count(candidate.text),
            )
        )

    return sorted(
        ranked,
        key=lambda item: (-item.score, -item.value, -item.surprise, item.candidate.id),
    )


def fit_to_budget(ranked: Sequence[ScoredCandidate], max_tokens: int) -> list[ScoredCandidate]:
    """Take the highest-value items that fit within the token budget."""

    selected: list[ScoredCandidate] = []
    tokens_used = 0
    for item in ranked:
        if item.token_count > max_tokens:
            continue
        if tokens_used + item.token_count > max_tokens:
            continue
        selected.append(item)
        tokens_used += item.token_count
    return selected


def select_context(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: Iterable[str],
    max_tokens: int,
) -> SelectionResult:
    """Rank and select candidates under a shared token budget."""

    ranked = rank_candidates(candidates, goal, current_context)
    selected = fit_to_budget(ranked, max_tokens)
    return SelectionResult(ranked=ranked, selected=selected, max_tokens=max_tokens)
