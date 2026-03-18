"""Core data models for the Psi0 prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    """A single candidate item that may enter the active context window."""

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate annotated with selection metrics."""

    candidate: Candidate
    score: float
    value: float
    surprise: float
    token_count: int


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Selection output with both ranked and budget-fitted candidates."""

    ranked: list[ScoredCandidate]
    selected: list[ScoredCandidate]
    max_tokens: int


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """Input for a candidate source query."""

    goal: str
    current_context: tuple[str, ...] = ()
    limit: int | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Fixture-style task definition used by the demo CLI and tests."""

    id: str
    goal: str
    current_context: list[str]
    max_tokens: int
    candidates: list[Candidate]
    metadata: dict[str, Any] = field(default_factory=dict)
