from pathlib import Path

from psi_loop.embedders import Embedder
from psi_loop.evaluation import evaluate_task
from psi_loop.forensics import build_task_forensics, render_task_forensics, trace_budget
from psi_loop.models import Candidate, ScoredCandidate
from psi_loop.sources import FixtureSource


def test_trace_budget_marks_budget_overflow_items():
    ranked = [
        ScoredCandidate(
            candidate=Candidate(id="first", text="one two three", source="a"),
            score=0.9,
            value=0.9,
            surprise=1.0,
            token_count=3,
        ),
        ScoredCandidate(
            candidate=Candidate(id="second", text="one two", source="b"),
            score=0.8,
            value=0.8,
            surprise=1.0,
            token_count=2,
        ),
    ]

    trace = trace_budget(ranked, max_tokens=3)

    assert [item.reason for item in trace] == ["selected", "would_exceed_budget"]
    assert trace[0].tokens_used_after == 3
    assert trace[1].tokens_used_before == 3


def test_build_task_forensics_returns_ranked_and_selected_stats():
    source = FixtureSource(Path(__file__).parent / "fixtures" / "benchmark_tasks.json")
    task = source.get_task("realistic_roadmap_planning")

    report = build_task_forensics(task, top_k=2)

    assert report.task_id == "realistic_roadmap_planning"
    assert report.gold_useful_candidates == ["novel_data_contracts"]
    assert report.gold_redundant_candidates == ["redundant_flaky_dashboards"]
    assert report.baseline.ranked_stats.count == 2
    assert report.baseline.selected_stats.count == 1
    assert report.psi0.selected[0].candidate.id == "novel_data_contracts"


def test_render_task_forensics_includes_gold_labels_and_stats():
    source = FixtureSource(Path(__file__).parent / "fixtures" / "benchmark_tasks.json")
    task = source.get_task("realistic_roadmap_planning")
    report = build_task_forensics(task, top_k=2)

    rendered = render_task_forensics(report, top_k=2)

    assert "Gold useful candidates:" in rendered
    assert "Gold redundant candidates:" in rendered
    assert "rank | baseline | psi0" in rendered
    assert "baseline top-2:" in rendered
    assert "psi0 selected:" in rendered


class RoadmapDenseEmbedder(Embedder):
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]


def test_roadmap_baseline_win_matches_published_dense_outcome():
    """Crafted embedder reproduces the published dense benchmark outcome for realistic_roadmap_planning: baseline wins, baseline selects novel_data_contracts, Psi0 selects unrelated_visual_refresh."""
    source = FixtureSource(Path(__file__).parent / "fixtures" / "benchmark_tasks.json")
    task = source.get_task("realistic_roadmap_planning")
    context = task.current_context[0]
    # 3D geometry: goal/context on (1,0,0); novel same as goal so baseline ranks it first; unrelated on (0,0,1) so high surprise and wins Psi0; redundant between.
    embedder = RoadmapDenseEmbedder(
        {
            task.goal: (1.0, 0.0, 0.0),
            context: (1.0, 0.0, 0.0),
            "Roadmap notes should say dashboards are flaky because upstream jobs fail silently.": (
                0.95,
                0.05,
                0.0,
            ),
            "Data contracts and freshness alerts are the concrete reliability investments missing from the roadmap.": (
                1.0,
                0.0,
                0.0,
            ),
            "Refresh chart colors for the analytics interface.": (0.0, 0.0, 1.0),
        }
    )

    report = build_task_forensics(task, embedder=embedder, top_k=3)
    task_result = evaluate_task(task, embedder=embedder)

    # Align with evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2.json: baseline wins, baseline selects novel_data_contracts, Psi0 selects unrelated_visual_refresh.
    assert report.baseline.selected[0].candidate.id == "novel_data_contracts"
    assert report.psi0.selected[0].candidate.id == "unrelated_visual_refresh"
    assert task_result["winner"] == "baseline"


def test_roadmap_task_forensics_with_default_embedder_exercises_plan_bonus():
    """Run realistic_roadmap_planning with default (Bow) embedder to exercise planning-bonus code path."""
    source = FixtureSource(Path(__file__).parent / "fixtures" / "benchmark_tasks.json")
    task = source.get_task("realistic_roadmap_planning")
    report = build_task_forensics(task, top_k=3)
    assert len(report.psi0.selected) >= 1
    assert len(report.baseline.selected) >= 1
    assert report.task_id == "realistic_roadmap_planning"
