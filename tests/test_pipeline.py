import json
from pathlib import Path

from psi_loop.baseline import select_context_baseline
from psi_loop.models import Candidate
from psi_loop.pipeline import select_context


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
