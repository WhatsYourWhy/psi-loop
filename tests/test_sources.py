import json
from pathlib import Path

from psi_loop.models import SourceRequest
from psi_loop.sources import FixtureSource


def test_fixture_source_lists_and_loads_tasks():
    source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")

    assert source.list_task_ids() == ["retry_backoff"]
    assert source.get_task("retry_backoff").goal.startswith("Design Python API client")


def test_fixture_source_fetch_respects_limit():
    source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")

    candidates = source.fetch(
        SourceRequest(goal="retry", task_id="retry_backoff", limit=2)
    )

    assert len(candidates) == 2
    assert candidates[0].id == "redundant_fixed_delay"


def test_bundled_sample_matches_test_fixture():
    packaged_fixture = (
        Path(__file__).parent.parent / "src" / "psi_loop" / "data" / "sample_tasks.json"
    )
    test_fixture = Path(__file__).parent / "fixtures" / "sample_tasks.json"

    packaged_payload = json.loads(packaged_fixture.read_text(encoding="utf-8"))
    test_payload = json.loads(test_fixture.read_text(encoding="utf-8"))

    assert packaged_payload == test_payload
