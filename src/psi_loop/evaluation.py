"""Evaluation helpers for comparing Psi0 against the similarity baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psi_loop.baseline import select_context_baseline
from psi_loop.embedders import BowEmbedder, Embedder
from psi_loop.models import ScoredCandidate, TaskDefinition
from psi_loop.pipeline import PsiLoop
from psi_loop.sources import FixtureSource


@dataclass(frozen=True, slots=True)
class SelectionMetrics:
    selected_ids: list[str]
    selected_token_total: int
    useful_hits: int
    redundant_hits: int
    useful_precision: float


def _selection_metrics(
    selected: list[ScoredCandidate],
    gold_useful: set[str],
    gold_redundant: set[str],
) -> SelectionMetrics:
    selected_ids = [item.candidate.id for item in selected]
    selected_token_total = sum(item.token_count for item in selected)
    useful_hits = sum(1 for item_id in selected_ids if item_id in gold_useful)
    redundant_hits = sum(1 for item_id in selected_ids if item_id in gold_redundant)
    useful_precision = useful_hits / len(selected_ids) if selected_ids else 0.0
    return SelectionMetrics(
        selected_ids=selected_ids,
        selected_token_total=selected_token_total,
        useful_hits=useful_hits,
        redundant_hits=redundant_hits,
        useful_precision=useful_precision,
    )


def _task_winner(psi_metrics: SelectionMetrics, baseline_metrics: SelectionMetrics) -> str:
    """Pick a winner while avoiding false positives when neither side is useful."""

    if psi_metrics.useful_hits > baseline_metrics.useful_hits:
        return "psi0"
    if psi_metrics.useful_hits < baseline_metrics.useful_hits:
        return "baseline"

    if psi_metrics.useful_hits == 0 and baseline_metrics.useful_hits == 0:
        return "tie"

    if psi_metrics.redundant_hits < baseline_metrics.redundant_hits:
        return "psi0"
    if psi_metrics.redundant_hits > baseline_metrics.redundant_hits:
        return "baseline"

    if psi_metrics.useful_precision > baseline_metrics.useful_precision:
        return "psi0"
    if psi_metrics.useful_precision < baseline_metrics.useful_precision:
        return "baseline"

    if psi_metrics.selected_token_total < baseline_metrics.selected_token_total:
        return "psi0"
    if psi_metrics.selected_token_total > baseline_metrics.selected_token_total:
        return "baseline"

    return "tie"


def evaluate_task(task: TaskDefinition, embedder: Embedder | None = None) -> dict[str, Any]:
    """Compare Psi0 and the similarity baseline on one benchmark task."""

    scorer = embedder if embedder is not None else BowEmbedder()
    psi_loop = PsiLoop(embedder=scorer)
    psi_result = psi_loop.select(
        goal=task.goal,
        current_context=task.current_context,
        max_tokens=task.max_tokens,
        candidates=task.candidates,
    )
    baseline_result = select_context_baseline(
        candidates=task.candidates,
        goal=task.goal,
        max_tokens=task.max_tokens,
        embedder=scorer,
    )

    gold_useful = set(task.metadata.get("gold_useful_candidates", []))
    gold_redundant = set(task.metadata.get("gold_redundant_candidates", []))

    psi_metrics = _selection_metrics(psi_result.selected, gold_useful, gold_redundant)
    baseline_metrics = _selection_metrics(
        baseline_result.selected,
        gold_useful,
        gold_redundant,
    )

    winner = _task_winner(psi_metrics, baseline_metrics)

    expected = task.metadata.get("expected_winner")
    return {
        "id": task.id,
        "category": task.metadata.get("category", "uncategorized"),
        "goal": task.goal,
        "notes": task.metadata.get("notes", ""),
        "expected_winner": expected,
        "winner": winner,
        "expected_match": winner == expected if expected is not None else None,
        "gold_useful_candidates": sorted(gold_useful),
        "gold_redundant_candidates": sorted(gold_redundant),
        "psi0": asdict(psi_metrics),
        "baseline": asdict(baseline_metrics),
    }


def summarize_results(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate task-level evaluation outputs."""

    totals = {"psi0": 0, "baseline": 0, "tie": 0}
    per_category: dict[str, dict[str, int | float]] = {}
    psi_useful_hits = 0
    baseline_useful_hits = 0
    psi_redundant_hits = 0
    baseline_redundant_hits = 0
    expected_match_count = 0

    for task in task_results:
        winner = task["winner"]
        totals[winner] += 1

        category = task["category"]
        if category not in per_category:
            per_category[category] = {"psi0": 0, "baseline": 0, "tie": 0, "tasks": 0}
        per_category[category][winner] += 1
        per_category[category]["tasks"] += 1

        psi_useful_hits += task["psi0"]["useful_hits"]
        baseline_useful_hits += task["baseline"]["useful_hits"]
        psi_redundant_hits += task["psi0"]["redundant_hits"]
        baseline_redundant_hits += task["baseline"]["redundant_hits"]
        if task["expected_match"] is True:
            expected_match_count += 1

    psi_win_rate = totals["psi0"] / len(task_results) if task_results else 0.0
    if (
        psi_win_rate >= 0.5
        and psi_useful_hits >= baseline_useful_hits + 4
        and psi_redundant_hits <= baseline_redundant_hits - 3
    ):
        decision = "proceed"
    elif psi_useful_hits > baseline_useful_hits or psi_redundant_hits < baseline_redundant_hits:
        decision = "refine_v"
    else:
        decision = "stop"

    return {
        "total_tasks": len(task_results),
        "wins": totals,
        "per_category": per_category,
        "psi0_useful_hits": psi_useful_hits,
        "baseline_useful_hits": baseline_useful_hits,
        "psi0_redundant_hits": psi_redundant_hits,
        "baseline_redundant_hits": baseline_redundant_hits,
        "expected_match_count": expected_match_count,
        "psi0_win_rate": psi_win_rate,
        "decision": decision,
    }


def run_benchmark(
    fixture_path: Path,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Run the benchmark fixture and return task-level plus aggregate results."""

    source = FixtureSource(fixture_path)
    tasks = source.tasks()
    task_results = [evaluate_task(task, embedder=embedder) for task in tasks]
    aggregate = summarize_results(task_results)
    return {
        "fixture_path": str(fixture_path),
        "embedder": type(embedder).__name__ if embedder is not None else "BowEmbedder",
        "task_results": task_results,
        "aggregate": aggregate,
    }


def write_results_json(results: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
