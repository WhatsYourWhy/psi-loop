from pathlib import Path

from psi_loop.embedders import Embedder, STEmbedder
from psi_loop.evaluation import describe_embedder, run_benchmark
from psi_loop.sources import FixtureSource


class DeterministicDenseEmbedder(Embedder):
    def embed(self, text: str) -> tuple[float, ...]:
        return (
            float(len(text)),
            float(sum(ord(char) for char in text) % 97),
            float(text.count(" ")),
        )


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


def test_run_benchmark_accepts_non_default_embedder():
    fixture = Path(__file__).parent / "fixtures" / "benchmark_tasks.json"

    results = run_benchmark(fixture, embedder=DeterministicDenseEmbedder())

    assert results["embedder"] == "DeterministicDenseEmbedder"
    assert results["embedder_metadata"] == {
        "backend": "custom",
        "class_name": "DeterministicDenseEmbedder",
        "model_name": None,
    }
    assert results["aggregate"]["total_tasks"] == 14


def test_describe_embedder_persists_dense_model_name():
    metadata = describe_embedder(STEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    assert metadata == {
        "backend": "dense",
        "class_name": "STEmbedder",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }
