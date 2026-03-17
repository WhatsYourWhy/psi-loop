"""CLI entry point for running the Psi0 prototype on fixture data."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path
from typing import Any

from psi_loop.baseline import select_context_baseline
from psi_loop.models import Candidate, SelectionResult
from psi_loop.pipeline import select_context


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


def _default_fixture_text() -> str:
    bundled_fixture = resources.files("psi_loop").joinpath("data/sample_tasks.json")
    return bundled_fixture.read_text(encoding="utf-8")


def _load_tasks(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        payload = json.loads(_default_fixture_text())
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not tasks:
        fixture_label = str(path) if path is not None else "bundled sample fixture"
        raise ValueError(f"No tasks were found in fixture: {fixture_label}")
    return tasks


def _pick_task(tasks: list[dict[str, Any]], task_id: str | None) -> dict[str, Any]:
    if task_id is None:
        return tasks[0]

    for task in tasks:
        if task["id"] == task_id:
            return task

    raise ValueError(f"Task '{task_id}' was not found in the fixture.")


def _to_candidates(raw_candidates: list[dict[str, Any]]) -> list[Candidate]:
    return [
        Candidate(
            id=item["id"],
            text=item["text"],
            source=item["source"],
            metadata=item.get("metadata", {}),
        )
        for item in raw_candidates
    ]


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

    tasks = _load_tasks(args.fixture)
    if args.list_tasks:
        for task in tasks:
            print(task["id"])
        return 0

    task = _pick_task(tasks, args.task)
    candidates = _to_candidates(task["candidates"])

    psi_result = select_context(
        candidates=candidates,
        goal=task["goal"],
        current_context=task.get("current_context", []),
        max_tokens=task.get("max_tokens", 64),
    )
    baseline_result = select_context_baseline(
        candidates=candidates,
        goal=task["goal"],
        max_tokens=task.get("max_tokens", 64),
    )

    print(f"Task: {task['id']}")
    print(f"Goal: {task['goal']}")
    if task.get("current_context"):
        print("Current context:")
        for entry in task["current_context"]:
            print(f"  - {entry}")

    _print_section("Psi0 selection", psi_result)
    _print_section("Baseline selection", baseline_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
