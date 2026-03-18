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
    """Build a structured forensic view for one task."""

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


def _format_candidate(item: ScoredCandidate | None) -> str:
    if item is None:
        return "-"
    return (
        f"{item.candidate.id} "
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


def render_task_forensics(report: TaskForensics, top_k: int) -> str:
    """Render a plain-text forensic report for one task."""

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
            f"{index + 1} | {_format_candidate(baseline_item)} | {_format_candidate(psi0_item)}"
        )

    for selector in (report.baseline, report.psi0):
        lines.append(f"{selector.name} budget trace:")
        for item in selector.budget_trace:
            lines.append(
                f"- {item.candidate.candidate.id}: reason={item.reason} "
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

    return "\n".join(lines)
