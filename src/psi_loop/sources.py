"""Candidate source protocols and fixture-backed implementations."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from psi_loop.models import Candidate, SourceRequest, TaskDefinition


class CandidateSource(Protocol):
    """Minimal source interface for retrieving candidate pools."""

    def fetch(self, request: SourceRequest) -> list[Candidate]:
        """Return candidate records for the given request."""


def _bundled_fixture_text() -> str:
    bundled_fixture = resources.files("psi_loop").joinpath("data/sample_tasks.json")
    return bundled_fixture.read_text(encoding="utf-8")


def _candidate_from_dict(item: dict[str, Any]) -> Candidate:
    return Candidate(
        id=item["id"],
        text=item["text"],
        source=item["source"],
        metadata=item.get("metadata", {}),
    )


def _task_from_dict(item: dict[str, Any]) -> TaskDefinition:
    known_fields = {"id", "goal", "current_context", "max_tokens", "candidates"}
    return TaskDefinition(
        id=item["id"],
        goal=item["goal"],
        current_context=item.get("current_context", []),
        max_tokens=item.get("max_tokens", 64),
        candidates=[_candidate_from_dict(candidate) for candidate in item["candidates"]],
        metadata={key: value for key, value in item.items() if key not in known_fields},
    )


class FixtureSource:
    """Source backed by a JSON fixture file or the bundled sample data."""

    def __init__(self, path: Path | None = None):
        self.path = path

    def _load_payload(self) -> dict[str, Any]:
        if self.path is None:
            return json.loads(_bundled_fixture_text())
        return json.loads(self.path.read_text(encoding="utf-8"))

    def tasks(self) -> list[TaskDefinition]:
        payload = self._load_payload()
        tasks = payload.get("tasks", [])
        if not tasks:
            fixture_label = str(self.path) if self.path is not None else "bundled sample fixture"
            raise ValueError(f"No tasks were found in fixture: {fixture_label}")
        return [_task_from_dict(task) for task in tasks]

    def list_task_ids(self) -> list[str]:
        return [task.id for task in self.tasks()]

    def get_task(self, task_id: str | None = None) -> TaskDefinition:
        tasks = self.tasks()
        if task_id is None:
            return tasks[0]
        for task in tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task '{task_id}' was not found in the fixture.")

    def fetch(self, request: SourceRequest) -> list[Candidate]:
        task = self.get_task(request.task_id)
        if request.limit is None:
            return list(task.candidates)
        return list(task.candidates[: request.limit])
