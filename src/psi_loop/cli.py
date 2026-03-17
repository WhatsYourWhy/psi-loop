"""CLI entry point for running the Psi0 prototype on fixture data."""

from __future__ import annotations

import argparse
from pathlib import Path

from psi_loop.baseline import baseline_score
from psi_loop.models import SelectionResult
from psi_loop.pipeline import PsiLoop
from psi_loop.sources import FixtureSource


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the psi-loop Psi0 prototype.")
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Path to a JSON fixture file containing task definitions.",
    )
    parser.add_argument(
        "--task",
        help="Task id to run. Defaults to the first task in the fixture.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List tasks in the selected fixture and exit.",
    )
    return parser


def _print_section(title: str, result: SelectionResult) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for item in result.selected:
        print(
            f"{item.candidate.id}: score={item.score:.3f} "
            f"value={item.value:.3f} surprise={item.surprise:.3f}"
        )
        print(f"  source={item.candidate.source}")
        print(f"  text={item.candidate.text}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    source = FixtureSource(args.fixture)
    if args.list_tasks:
        for task_id in source.list_task_ids():
            print(task_id)
        return 0

    task = source.get_task(args.task)
    psi_loop = PsiLoop(source=source)
    baseline_loop = PsiLoop(source=source, scorer=baseline_score)

    psi_result = psi_loop.select(
        goal=task.goal,
        current_context=task.current_context,
        max_tokens=task.max_tokens,
        candidates=task.candidates,
        task_id=task.id,
    )
    baseline_result = baseline_loop.select(
        goal=task.goal,
        current_context=task.current_context,
        max_tokens=task.max_tokens,
        candidates=task.candidates,
        task_id=task.id,
    )

    print(f"Task: {task.id}")
    print(f"Goal: {task.goal}")
    if task.current_context:
        print("Current context:")
        for entry in task.current_context:
            print(f"  - {entry}")

    _print_section("Psi0 selection", psi_result)
    _print_section("Baseline selection", baseline_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
