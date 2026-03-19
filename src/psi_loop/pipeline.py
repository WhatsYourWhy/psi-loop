"""Budgeted selection pipeline for Psi0-ranked candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from psi_loop.embedders import Embedder
from psi_loop.models import Candidate, ScoredCandidate, SelectionResult, SourceRequest
from psi_loop.sources import CandidateSource
from psi_loop.scoring import psi_0, tokenize

CandidateScorer = Callable[[str, str, Iterable[str], Embedder | None], tuple[float, float, float]]

# When two scores differ by less than this, rank by value (higher V first) before budget packing.
NEAR_TIE_EPSILON = 1e-2


def _token_count(text: str) -> int:
    return len(tokenize(text))


def rank_candidates(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: Iterable[str],
    scorer: CandidateScorer = psi_0,
    embedder: Embedder | None = None,
) -> list[ScoredCandidate]:
    """Rank candidates with Psi0 and preserve useful scoring detail.

    When two scores differ by less than NEAR_TIE_EPSILON, the candidate with higher
    value is ranked first (near-tie V-priority) so budget packing prefers usefulness
    over novelty in close score contests.
    """

    ranked: list[ScoredCandidate] = []
    for candidate in candidates:
        score, value, surprise = scorer(
            candidate.text,
            goal,
            current_context,
            embedder,
        )
        ranked.append(
            ScoredCandidate(
                candidate=candidate,
                score=score,
                value=value,
                surprise=surprise,
                token_count=_token_count(candidate.text),
            )
        )

    def sort_key(item: ScoredCandidate) -> tuple[float, float, float, str]:
        score_bucket = (item.score // NEAR_TIE_EPSILON) * NEAR_TIE_EPSILON
        return (-score_bucket, -item.value, -item.surprise, item.candidate.id)

    return sorted(ranked, key=sort_key)


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
    embedder: Embedder | None = None,
) -> SelectionResult:
    """Rank and select candidates under a shared token budget."""

    ranked = rank_candidates(
        candidates,
        goal,
        current_context,
        scorer=psi_0,
        embedder=embedder,
    )
    selected = fit_to_budget(ranked, max_tokens)
    return SelectionResult(ranked=ranked, selected=selected, max_tokens=max_tokens)


def select_with_scorer(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: Iterable[str],
    max_tokens: int,
    scorer: CandidateScorer,
    embedder: Embedder | None = None,
) -> SelectionResult:
    """Shared selection entrypoint for pluggable scoring strategies."""

    ranked = rank_candidates(
        candidates,
        goal,
        current_context,
        scorer=scorer,
        embedder=embedder,
    )
    selected = fit_to_budget(ranked, max_tokens)
    return SelectionResult(ranked=ranked, selected=selected, max_tokens=max_tokens)


class PsiLoop:
    """Thin orchestration shell over candidate fetching and ranking."""

    def __init__(
        self,
        source: CandidateSource | None = None,
        embedder: Embedder | None = None,
        scorer: CandidateScorer = psi_0,
    ):
        self.source = source
        self.embedder = embedder
        self.scorer = scorer

    def select(
        self,
        goal: str,
        current_context: Iterable[str],
        max_tokens: int,
        candidates: Sequence[Candidate] | None = None,
        fetch_k: int | None = None,
        task_id: str | None = None,
    ) -> SelectionResult:
        """Select context from provided candidates or a configured source."""

        context_items = tuple(current_context)
        if candidates is None:
            if self.source is None:
                raise ValueError("PsiLoop requires either candidates or a configured source.")
            request = SourceRequest(
                goal=goal,
                current_context=context_items,
                limit=fetch_k,
                task_id=task_id,
            )
            candidates = self.source.fetch(request)

        return select_with_scorer(
            candidates=candidates,
            goal=goal,
            current_context=context_items,
            max_tokens=max_tokens,
            scorer=self.scorer,
            embedder=self.embedder,
        )
