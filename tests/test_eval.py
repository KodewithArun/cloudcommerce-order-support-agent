"""Automated tests running the conversation evaluation suite."""

from __future__ import annotations

from app.eval import SCENARIOS, ConversationEvaluator


def test_all_conversation_scenarios_pass_offline() -> None:
    """Run all 9 evaluation scenarios offline and assert 100% pass rate."""
    evaluator = ConversationEvaluator(live=False)
    results = evaluator.run_all(SCENARIOS)

    assert len(results) >= 8, "Requirement specifies at least 8 test conversations"

    refusals = sum(1 for r in results if any(t.is_refusal for t in r.scenario.turns))
    assert refusals >= 2, "Requirement specifies at least 2 refusals"

    failed = [r for r in results if not r.passed]
    if failed:
        reasons = [f"{r.scenario.title}: {r.failure_summary}" for r in failed]
        raise AssertionError(f"Scenarios failed: {reasons}")

    assert all(r.passed for r in results)
