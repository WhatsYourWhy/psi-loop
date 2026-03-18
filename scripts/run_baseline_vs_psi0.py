"""Run the benchmark comparison between Psi0 and the similarity baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from psi_loop.embedders import BowEmbedder, STEmbedder
from psi_loop.evaluation import run_benchmark, write_results_json


def default_output_path_for_backend(backend: str) -> Path:
    return Path(f"evaluation_results_baseline_vs_psi0_{backend}.json")


def resolve_json_output_path(json_out: Path | None, backend: str) -> Path:
    return json_out if json_out is not None else default_output_path_for_backend(backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Psi0 vs baseline benchmark.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/benchmark_tasks.json"),
        help="Path to the benchmark fixture JSON.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the structured JSON results.",
    )
    parser.add_argument(
        "--backend",
        choices=("bow", "dense"),
        default="bow",
        help="Embedder backend to use for the benchmark run.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Dense model name to use when --backend dense is selected.",
    )
    return parser


def print_summary(results: dict) -> None:
    aggregate = results["aggregate"]
    embedder_metadata = results["embedder_metadata"]
    print("Psi0 vs baseline evaluation")
    print("==========================")
    print(f"Fixture: {results['fixture_path']}")
    print(
        f"Embedder: {results['embedder']} "
        f"(backend={embedder_metadata['backend']}, "
        f"model={embedder_metadata['model_name'] or 'n/a'})"
    )
    print()
    print(
        "Wins: "
        f"psi0={aggregate['wins']['psi0']} "
        f"baseline={aggregate['wins']['baseline']} "
        f"tie={aggregate['wins']['tie']}"
    )
    print(
        "Useful hits: "
        f"psi0={aggregate['psi0_useful_hits']} "
        f"baseline={aggregate['baseline_useful_hits']}"
    )
    print(
        "Redundant hits: "
        f"psi0={aggregate['psi0_redundant_hits']} "
        f"baseline={aggregate['baseline_redundant_hits']}"
    )
    print(f"Expected-match count: {aggregate['expected_match_count']}/{aggregate['total_tasks']}")
    print(f"Decision: {aggregate['decision']}")
    print()
    print("Per-task results")
    print("----------------")
    for task in results["task_results"]:
        print(
            f"{task['id']}: winner={task['winner']} "
            f"expected={task['expected_winner']} "
            f"psi0={task['psi0']['selected_ids']} "
            f"baseline={task['baseline']['selected_ids']}"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.backend == "dense":
        embedder = STEmbedder(model_name=args.model_name)
    else:
        embedder = BowEmbedder()

    results = run_benchmark(args.fixture, embedder=embedder)
    output_path = resolve_json_output_path(args.json_out, args.backend)
    write_results_json(results, output_path)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
