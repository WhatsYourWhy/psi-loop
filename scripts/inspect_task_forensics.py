"""Inspect one benchmark task with side-by-side baseline and Psi0 traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from psi_loop.embedders import BowEmbedder, STEmbedder
from psi_loop.forensics import build_task_forensics, render_task_forensics
from psi_loop.sources import FixtureSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one benchmark task in detail.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/benchmark_tasks.json"),
        help="Path to the benchmark fixture JSON.",
    )
    parser.add_argument(
        "--task-id",
        default="realistic_roadmap_planning",
        help="Task id to inspect.",
    )
    parser.add_argument(
        "--backend",
        choices=("bow", "dense"),
        default="bow",
        help="Embedder backend to use for the forensic run.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Dense model name to use when --backend dense is selected.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ranked candidates to print per selector.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = FixtureSource(args.fixture)
    task = source.get_task(args.task_id)
    if args.backend == "dense":
        embedder = STEmbedder(model_name=args.model_name)
    else:
        embedder = BowEmbedder()

    report = build_task_forensics(task, embedder=embedder, top_k=args.top_k)
    print(render_task_forensics(report, top_k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
