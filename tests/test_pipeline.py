import json
from pathlib import Path

from psi_loop.baseline import select_context_baseline
from psi_loop.embedders import Embedder
from psi_loop.models import Candidate
from psi_loop.pipeline import PsiLoop, select_context
from psi_loop.sources import FixtureSource


def _load_task(task_id: str) -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_tasks.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return next(task for task in payload["tasks"] if task["id"] == task_id)


def _build_candidates(task: dict) -> list[Candidate]:
    return [
        Candidate(id=item["id"], text=item["text"], source=item["source"])
        for item in task["candidates"]
    ]


def test_psi0_prefers_novel_relevant_candidate():
    task = _load_task("retry_backoff")
    candidates = _build_candidates(task)

    result = select_context(
        candidates=candidates,
        goal=task["goal"],
        current_context=task["current_context"],
        max_tokens=task["max_tokens"],
    )

    assert result.ranked[0].candidate.id == "novel_backoff_jitter"
    assert [item.candidate.id for item in result.selected] == ["novel_backoff_jitter"]


def test_baseline_prefers_redundant_candidate():
    task = _load_task("retry_backoff")
    candidates = _build_candidates(task)

    result = select_context_baseline(
        candidates=candidates,
        goal=task["goal"],
        max_tokens=task["max_tokens"],
    )

    assert result.ranked[0].candidate.id == "redundant_fixed_delay"


def test_budgeted_selection_skips_items_that_do_not_fit():
    candidates = [
        Candidate(id="large", text="one two three four five six", source="a"),
        Candidate(id="small", text="one two", source="b"),
        Candidate(id="tiny", text="one", source="c"),
    ]

    result = select_context(
        candidates=candidates,
        goal="one two three four",
        current_context=[],
        max_tokens=3,
    )

    selected_ids = [item.candidate.id for item in result.selected]
    assert "large" not in selected_ids
    assert selected_ids


def test_psiloop_fetches_from_source_when_candidates_are_not_provided():
    source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")
    loop = PsiLoop(source=source)
    task = source.get_task("retry_backoff")

    result = loop.select(
        goal=task.goal,
        current_context=task.current_context,
        max_tokens=task.max_tokens,
        task_id=task.id,
    )

    assert result.ranked[0].candidate.id == "novel_backoff_jitter"


def test_weighted_v_prefers_mechanism_candidate_over_generic_goal_shell():
    candidates = [
        Candidate(id="generic", text="Design Python API client logic", source="a"),
        Candidate(id="mechanism", text="Use backoff jitter retry", source="b"),
    ]

    result = select_context(
        candidates=candidates,
        goal="Design Python API client retry logic with exponential backoff and jitter",
        current_context=[],
        max_tokens=10,
    )

    assert result.ranked[0].candidate.id == "mechanism"


class OpposingDenseEmbedder(Embedder):
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]


def test_baseline_does_not_select_negative_dense_scores():
    candidates = [
        Candidate(id="anti", text="anti", source="a"),
        Candidate(id="also_anti", text="also_anti", source="b"),
    ]
    embedder = OpposingDenseEmbedder(
        {
            "goal": (1.0, 0.0),
            "anti": (-1.0, 0.0),
            "also_anti": (-0.5, 0.0),
        }
    )

    result = select_context_baseline(
        candidates=candidates,
        goal="goal",
        max_tokens=10,
        embedder=embedder,
    )

    assert [item.candidate.id for item in result.ranked] == ["also_anti", "anti"]
    assert result.selected == []
