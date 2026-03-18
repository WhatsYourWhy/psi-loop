from pathlib import Path

from psi_loop.evaluation import run_benchmark
from psi_loop.sources import FixtureSource


def test_benchmark_fixture_loads_with_metadata():
    fixture = Path(__file__).parent / "fixtures" / "benchmark_tasks.json"
    source = FixtureSource(fixture)

    tasks = source.tasks()

    assert len(tasks) == 14
    assert tasks[0].metadata["category"] == "synthetic_redundancy"
    assert tasks[-1].metadata["category"] == "realistic_knowledge_work"


def test_run_benchmark_returns_expected_structure():
    fixture = Path(__file__).parent / "fixtures" / "benchmark_tasks.json"

    results = run_benchmark(fixture)

    assert results["aggregate"]["total_tasks"] == 14
    assert "task_results" in results
    assert results["task_results"][0]["winner"] in {"psi0", "baseline", "tie"}
    assert "psi0" in results["task_results"][0]
    assert "baseline" in results["task_results"][0]
