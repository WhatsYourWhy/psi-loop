import json
from pathlib import Path

from psi_loop.baseline import select_context_baseline
from psi_loop.embedders import Embedder
from psi_loop.models import Candidate
from psi_loop.pipeline import PsiLoop, rank_candidates, select_context
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


def test_near_tie_v_priority_ranks_higher_value_first():
    """When two candidates have scores within NEAR_TIE_EPSILON, higher value is rank-1."""
    candidates = [
        Candidate(id="low_v", text="low_v_note", source="a"),
        Candidate(id="high_v", text="high_v_note", source="b"),
    ]
    # Scores 0.232 vs 0.231 (within 0.01); high_v has value 0.41, low_v has 0.30.
    scores_by_text = {
        "low_v_note": (0.232, 0.30, 0.8),
        "high_v_note": (0.231, 0.41, 0.5),
    }

    def scorer(text: str, goal: str, context, embedder):
        return scores_by_text[text]

    ranked = rank_candidates(
        candidates,
        goal="goal",
        current_context=[],
        scorer=scorer,
    )
    assert ranked[0].candidate.id == "high_v"
    assert ranked[0].value == 0.41
    assert ranked[1].candidate.id == "low_v"
    assert ranked[1].value == 0.30


def test_near_tie_large_score_gap_still_ranks_by_score():
    """When scores differ by more than epsilon, higher score wins (no tie)."""
    candidates = [
        Candidate(id="low_score", text="low_score_note", source="a"),
        Candidate(id="high_score", text="high_score_note", source="b"),
    ]
    # Scores 0.25 vs 0.10 (gap 0.15 > 0.01); low_score has higher value but lower score.
    scores_by_text = {
        "low_score_note": (0.10, 0.9, 0.1),
        "high_score_note": (0.25, 0.5, 0.5),
    }

    def scorer(text: str, goal: str, context, embedder):
        return scores_by_text[text]

    ranked = rank_candidates(
        candidates,
        goal="goal",
        current_context=[],
        scorer=scorer,
    )
    assert ranked[0].candidate.id == "high_score"
    assert ranked[0].score == 0.25


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
