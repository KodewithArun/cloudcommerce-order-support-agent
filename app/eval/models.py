"""Data models for conversation scenarios and evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single turn in an evaluation scenario."""

    message: str
    expected_tool: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    is_refusal: bool = False
    is_approval: bool = False


@dataclass
class ConversationScenario:
    """A multi-turn conversation scenario to evaluate."""

    scenario_id: str
    title: str
    customer_id: str
    description: str
    turns: list[Turn]


@dataclass
class TurnResult:
    """Evaluation result for a single turn."""

    turn_index: int
    user_message: str
    agent_response: str
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)
    tool_called: str | None = None
    duration_ms: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Evaluation result for an entire multi-turn scenario."""

    scenario: ConversationScenario
    passed: bool
    turn_results: list[TurnResult]
    total_duration_ms: float
    total_tokens: int
    failure_summary: list[str] = field(default_factory=list)
