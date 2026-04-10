"""Task-level forensic helpers for comparing baseline and Psi0 selections."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from psi_loop.baseline import select_context_baseline
from psi_loop.embedders import Embedder
from psi_loop.models import ScoredCandidate, SelectionResult, TaskDefinition
from psi_loop.pipeline import select_context

BudgetReason = Literal["selected", "too_large", "would_exceed_budget"]


@dataclass(frozen=True, slots=True)
class BudgetTraceItem:
    candidate: ScoredCandidate
    reason: BudgetReason
    tokens_used_before: int
    tokens_used_after: int


@dataclass(frozen=True, slots=True)
class ContributionStats:
    count: int
    mean_score: float
    mean_value: float
    mean_surprise: float


@dataclass(frozen=True, slots=True)
class SelectorForensics:
    name: str
    ranked: list[ScoredCandidate]
    selected: list[ScoredCandidate]
    budget_trace: list[BudgetTraceItem]
    ranked_stats: ContributionStats
    selected_stats: ContributionStats


@dataclass(frozen=True, slots=True)
class TaskForensics:
    task_id: str
    goal: str
    current_context: list[str]
    max_tokens: int
    gold_useful_candidates: list[str]
    gold_redundant_candidates: list[str]
    baseline: SelectorForensics
    psi0: SelectorForensics


def trace_budget(ranked: list[ScoredCandidate], max_tokens: int) -> list[BudgetTraceItem]:
    """Replay budget fitting to explain which items were selected or dropped."""

    trace: list[BudgetTraceItem] = []
    tokens_used = 0
    for item in ranked:
        tokens_used_before = tokens_used
        if item.token_count > max_tokens:
            trace.append(
                BudgetTraceItem(
                    candidate=item,
                    reason="too_large",
                    tokens_used_before=tokens_used_before,
                    tokens_used_after=tokens_used_before,
                )
            )
            continue
        if tokens_used + item.token_count > max_tokens:
            trace.append(
                BudgetTraceItem(
                    candidate=item,
                    reason="would_exceed_budget",
                    tokens_used_before=tokens_used_before,
                    tokens_used_after=tokens_used_before,
                )
            )
            continue

        tokens_used += item.token_count
        trace.append(
            BudgetTraceItem(
                candidate=item,
                reason="selected",
                tokens_used_before=tokens_used_before,
                tokens_used_after=tokens_used,
            )
        )
    return trace


def contribution_stats(items: list[ScoredCandidate]) -> ContributionStats:
    """Summarize score components over a candidate slice."""

    if not items:
        return ContributionStats(
            count=0,
            mean_score=0.0,
            mean_value=0.0,
            mean_surprise=0.0,
        )

    return ContributionStats(
        count=len(items),
        mean_score=fmean(item.score for item in items),
        mean_value=fmean(item.value for item in items),
        mean_surprise=fmean(item.surprise for item in items),
    )


def _selector_forensics(
    name: str,
    result: SelectionResult,
    max_tokens: int,
    top_k: int,
) -> SelectorForensics:
    ranked_slice = result.ranked[:top_k]
    return SelectorForensics(
        name=name,
        ranked=result.ranked,
        selected=result.selected,
        budget_trace=trace_budget(result.ranked, max_tokens),
        ranked_stats=contribution_stats(ranked_slice),
        selected_stats=contribution_stats(result.selected),
    )


def build_task_forensics(
    task: TaskDefinition,
    embedder: Embedder | None = None,
    top_k: int = 5,
) -> TaskForensics:
    """Build a structured forensic view for one task.

    Uses iterative=False so that the budget trace (which replays selection over
    the initial ranked list) stays consistent with the reported selected set.
    Iterative selection interleaves scoring and selection in a way that cannot
    be faithfully replayed by a post-hoc budget trace over a static ranking.
    """

    baseline_result = select_context_baseline(
        candidates=task.candidates,
        goal=task.goal,
        max_tokens=task.max_tokens,
        embedder=embedder,
    )
    psi0_result = select_context(
        candidates=task.candidates,
        goal=task.goal,
        current_context=task.current_context,
        max_tokens=task.max_tokens,
        embedder=embedder,
        iterative=False,
    )
    return TaskForensics(
        task_id=task.id,
        goal=task.goal,
        current_context=task.current_context,
        max_tokens=task.max_tokens,
        gold_useful_candidates=sorted(task.metadata.get("gold_useful_candidates", [])),
        gold_redundant_candidates=sorted(task.metadata.get("gold_redundant_candidates", [])),
        baseline=_selector_forensics("baseline", baseline_result, task.max_tokens, top_k),
        psi0=_selector_forensics("psi0", psi0_result, task.max_tokens, top_k),
    )


def _format_candidate(
    item: ScoredCandidate | None,
    gold_useful: frozenset[str] | None = None,
    gold_redundant: frozenset[str] | None = None,
) -> str:
    if item is None:
        return "-"
    label = ""
    if gold_useful and item.candidate.id in gold_useful:
        label = " [gold useful]"
    elif gold_redundant and item.candidate.id in gold_redundant:
        label = " [gold redundant]"
    return (
        f"{item.candidate.id}{label} "
        f"score={item.score:.4f} "
        f"value={item.value:.4f} "
        f"surprise={item.surprise:.4f} "
        f"tokens={item.token_count}"
    )


def _format_stats(name: str, stats: ContributionStats) -> str:
    return (
        f"{name}: "
        f"count={stats.count} "
        f"mean_score={stats.mean_score:.4f} "
        f"mean_value={stats.mean_value:.4f} "
        f"mean_surprise={stats.mean_surprise:.4f}"
    )


def _diagnosis_block(report: TaskForensics) -> str:
    """One-line summary and diagnosis: Psi0 vs baseline rank-1, gold position, and error type."""
    lines: list[str] = []
    psi0_r1 = report.psi0.ranked[0] if report.psi0.ranked else None
    baseline_r1 = report.baseline.ranked[0] if report.baseline.ranked else None
    if psi0_r1:
        lines.append(
            f"Psi0 rank-1: {psi0_r1.candidate.id} "
            f"value={psi0_r1.value:.4f} surprise={psi0_r1.surprise:.4f}"
        )
    if baseline_r1:
        lines.append(
            f"Baseline rank-1: {baseline_r1.candidate.id} "
            f"value={baseline_r1.value:.4f} surprise={baseline_r1.surprise:.4f}"
        )
    gold_useful_set = set(report.gold_useful_candidates)
    for gold_id in report.gold_useful_candidates:
        psi0_rank = next(
            (i + 1 for i, sc in enumerate(report.psi0.ranked) if sc.candidate.id == gold_id),
            None,
        )
        baseline_rank = next(
            (i + 1 for i, sc in enumerate(report.baseline.ranked) if sc.candidate.id == gold_id),
            None,
        )
        if psi0_rank is not None:
            sc = report.psi0.ranked[psi0_rank - 1]
            lines.append(
                f"Gold useful {gold_id} in Psi0: rank={psi0_rank} "
                f"value={sc.value:.4f} surprise={sc.surprise:.4f}"
            )
        else:
            lines.append(f"Gold useful {gold_id} in Psi0: not in ranked list")
        if baseline_rank is not None:
            sc = report.baseline.ranked[baseline_rank - 1]
            lines.append(
                f"Gold useful {gold_id} in baseline: rank={baseline_rank} "
                f"value={sc.value:.4f} surprise={sc.surprise:.4f}"
            )
        else:
            lines.append(f"Gold useful {gold_id} in baseline: not in ranked list")

    # Diagnosis: Low V on gold | High H on Psi0 winner | Budget | Other
    diagnosis = "Other"
    if report.psi0.selected and report.baseline.selected:
        psi0_selected_ids = {sc.candidate.id for sc in report.psi0.selected}
        baseline_selected_ids = {sc.candidate.id for sc in report.baseline.selected}
        gold_in_psi0_selected = bool(gold_useful_set & psi0_selected_ids)
        gold_in_baseline_selected = bool(gold_useful_set & baseline_selected_ids)
        psi0_trace_by_id = {item.candidate.candidate.id: item for item in report.psi0.budget_trace}
        for gold_id in report.gold_useful_candidates:
            if gold_id in psi0_trace_by_id:
                reason = psi0_trace_by_id[gold_id].reason
                if reason in ("too_large", "would_exceed_budget") and not gold_in_psi0_selected:
                    diagnosis = "Budget"
                    break
        if diagnosis == "Other" and not gold_in_psi0_selected and psi0_r1 and gold_useful_set:
            gold_sc = next(
                (sc for sc in report.psi0.ranked if sc.candidate.id in gold_useful_set),
                None,
            )
            if gold_sc and psi0_r1.value <= gold_sc.value and psi0_r1.surprise > gold_sc.surprise:
                diagnosis = "High H on Psi0 winner"
            elif gold_sc and gold_sc.value < psi0_r1.value:
                diagnosis = "Low V on gold"
    lines.append(f"Diagnosis: {diagnosis}")
    return "\n".join(lines)


def render_task_forensics(report: TaskForensics, top_k: int) -> str:
    """Render a plain-text forensic report for one task."""

    gold_useful = frozenset(report.gold_useful_candidates)
    gold_redundant = frozenset(report.gold_redundant_candidates)

    lines = [
        f"Task forensic: {report.task_id}",
        f"Goal: {report.goal}",
        f"Budget: {report.max_tokens} tokens",
        "Current context:",
    ]
    lines.extend(f"- {item}" for item in report.current_context)
    lines.append("Gold useful candidates:")
    lines.extend(f"- {item}" for item in report.gold_useful_candidates)
    lines.append("Gold redundant candidates:")
    lines.extend(f"- {item}" for item in report.gold_redundant_candidates)
    lines.append("Top-ranked candidates (baseline vs psi0):")
    lines.append("rank | baseline | psi0")

    for index in range(top_k):
        baseline_item = report.baseline.ranked[index] if index < len(report.baseline.ranked) else None
        psi0_item = report.psi0.ranked[index] if index < len(report.psi0.ranked) else None
        lines.append(
            f"{index + 1} | {_format_candidate(baseline_item, gold_useful, gold_redundant)} | "
            f"{_format_candidate(psi0_item, gold_useful, gold_redundant)}"
        )

    for selector in (report.baseline, report.psi0):
        lines.append(f"{selector.name} budget trace:")
        for item in selector.budget_trace:
            label = ""
            if item.candidate.candidate.id in gold_useful:
                label = " [gold useful]"
            elif item.candidate.candidate.id in gold_redundant:
                label = " [gold redundant]"
            lines.append(
                f"- {item.candidate.candidate.id}{label}: reason={item.reason} "
                f"score={item.candidate.score:.4f} "
                f"value={item.candidate.value:.4f} "
                f"surprise={item.candidate.surprise:.4f} "
                f"tokens={item.candidate.token_count} "
                f"budget={item.tokens_used_before}->{item.tokens_used_after}"
            )
        lines.append(
            f"{selector.name} selected ids: "
            f"{[item.candidate.id for item in selector.selected]}"
        )
        lines.append(_format_stats(f"{selector.name} top-{top_k}", selector.ranked_stats))
        lines.append(_format_stats(f"{selector.name} selected", selector.selected_stats))

    lines.append("")
    lines.append(_diagnosis_block(report))
    return "\n".join(lines)
