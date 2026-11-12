"""Evaluation package for CloudCommerce conversation testing."""

from app.eval.models import (
    ConversationScenario,
    ScenarioResult,
    Turn,
    TurnResult,
)
from app.eval.runner import ConversationEvaluator
from app.eval.scenarios import SCENARIOS

__all__ = [
    "ConversationEvaluator",
    "ConversationScenario",
    "SCENARIOS",
    "ScenarioResult",
    "Turn",
    "TurnResult",
]

