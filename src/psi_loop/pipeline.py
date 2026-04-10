"""Budgeted selection pipeline for Psi0-ranked candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from psi_loop.embedders import Embedder
from psi_loop.models import Candidate, ScoredCandidate, SelectionResult, SourceRequest
from psi_loop.sources import CandidateSource
from psi_loop.scoring import psi_0, tokenize

CandidateScorer = Callable[[str, str, Iterable[str], Embedder | None], tuple[float, float, float]]

# Scores within the same NEAR_TIE_EPSILON-wide floor-division bucket are treated as near-ties
# and ranked by value (higher V first) before budget packing.  This is implemented as floor
# division, so the guarantee is per-bucket, not per-pair: two scores that straddle a bucket
# boundary may differ by less than NEAR_TIE_EPSILON yet still land in different buckets.
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
    """Take the highest-scoring items that fit within the token budget.

    Scores are fixed before selection begins — selected candidates do not
    update the context centroid.  Use select_context (iterative=True) when
    you want each selection to suppress redundant subsequent picks.
    """

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


def _select_iterative(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: list[str],
    max_tokens: int,
    scorer: CandidateScorer,
    embedder: Embedder | None,
) -> list[ScoredCandidate]:
    """Greedy iterative selection: re-score remaining candidates after each pick.

    Each selected candidate is appended to the running context before the next
    round of scoring, so already-selected content suppresses redundant picks —
    not just the original current_context.  This makes the redundancy guarantee
    hold within the selected set, not just against pre-existing context.
    """

    running_context = list(current_context)
    remaining: list[Candidate] = list(candidates)
    selected: list[ScoredCandidate] = []
    tokens_used = 0

    while remaining:
        # Score all remaining candidates against the running context.
        scored: list[ScoredCandidate] = []
        for candidate in remaining:
            score, value, surprise = scorer(candidate.text, goal, running_context, embedder)
            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    score=score,
                    value=value,
                    surprise=surprise,
                    token_count=_token_count(candidate.text),
                )
            )

        scored.sort(key=lambda x: (-x.score, -x.value, x.candidate.id))

        # Pick the highest-scoring candidate that fits in the remaining budget.
        picked: ScoredCandidate | None = None
        for item in scored:
            if item.token_count <= max_tokens and tokens_used + item.token_count <= max_tokens:
                picked = item
                break

        if picked is None:
            break  # Nothing fits — done.

        selected.append(picked)
        tokens_used += picked.token_count
        running_context.append(picked.candidate.text)
        remaining = [c for c in remaining if c.id != picked.candidate.id]

    return selected


def select_context(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: Iterable[str],
    max_tokens: int,
    embedder: Embedder | None = None,
    iterative: bool = True,
) -> SelectionResult:
    """Rank and select candidates under a shared token budget.

    When iterative=True (default), each selected candidate updates the running
    context before scoring the next round, so selected items suppress redundant
    subsequent picks.  Set iterative=False to use the original single-pass
    fit_to_budget behaviour.
    """

    context_items = list(current_context)

    # Always produce a full initial ranking for inspection / forensics.
    ranked = rank_candidates(
        candidates,
        goal,
        context_items,
        scorer=psi_0,
        embedder=embedder,
    )

    if iterative:
        selected = _select_iterative(candidates, goal, context_items, max_tokens, psi_0, embedder)
    else:
        selected = fit_to_budget(ranked, max_tokens)

    return SelectionResult(ranked=ranked, selected=selected, max_tokens=max_tokens)


def select_with_scorer(
    candidates: Sequence[Candidate],
    goal: str,
    current_context: Iterable[str],
    max_tokens: int,
    scorer: CandidateScorer,
    embedder: Embedder | None = None,
    iterative: bool = True,
) -> SelectionResult:
    """Shared selection entrypoint for pluggable scoring strategies."""

    context_items = list(current_context)
    ranked = rank_candidates(
        candidates,
        goal,
        context_items,
        scorer=scorer,
        embedder=embedder,
    )

    if iterative:
        selected = _select_iterative(candidates, goal, context_items, max_tokens, scorer, embedder)
    else:
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
        iterative: bool = True,
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
            iterative=iterative,
        )
