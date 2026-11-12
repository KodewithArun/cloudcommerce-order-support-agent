"""Evaluation runner executing test conversations and validating expected outcomes."""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent
from app.chat.service import ChatService
from app.eval.models import ConversationScenario, ScenarioResult, Turn, TurnResult
from app.eval.scenarios import SCENARIOS
from tests.test_agent import ScriptedChatModel, _tool_call


def _build_scripted_model_for_scenario(scenario_id: str) -> ScriptedChatModel:
    """Return deterministic scripted model responses for offline evaluation."""
    meta = {"input_tokens": 45, "output_tokens": 25, "total_tokens": 70}
    responses_map: dict[str, list[AIMessage]] = {
        "scenario_01_order_status": [
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A100"})], usage_metadata=meta),
            AIMessage(content="Your order A100 (Wireless Headphones) was delivered on 2026-11-01.", usage_metadata=meta),
        ],
        "scenario_02_followup_cancel": [
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A102"})], usage_metadata=meta),
            AIMessage(content="Order A102 is currently processing.", usage_metadata=meta),
            AIMessage(content="", tool_calls=[_tool_call("cancel_order", {"order_id": "A102"})], usage_metadata=meta),
            AIMessage(content="Order A102 has been cancelled. Any refund will take 5-7 business days.", usage_metadata=meta),
        ],
        "scenario_03_followup_other_order": [
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A100"})], usage_metadata=meta),
            AIMessage(content="Order A100 was delivered.", usage_metadata=meta),
            AIMessage(content="", tool_calls=[_tool_call("list_orders", {})], usage_metadata=meta),
            AIMessage(content="Your other order A101 (Phone Case) is shipped, and A102 is processing.", usage_metadata=meta),
        ],
        "scenario_04_cross_customer_refusal": [
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "B200"})], usage_metadata=meta),
            AIMessage(content="No order B200 was found for your customer account.", usage_metadata=meta),
        ],
        "scenario_05_cancel_delivered_refusal": [
            AIMessage(content="", tool_calls=[_tool_call("cancel_order", {"order_id": "A100"})], usage_metadata=meta),
            AIMessage(content="Order A100 is delivered and cannot be cancelled per our cancellation policy.", usage_metadata=meta),
        ],
        "scenario_06_refund_expired_window_refusal": [
            AIMessage(content="", tool_calls=[_tool_call("request_refund", {"order_id": "B200"})], usage_metadata=meta),
            AIMessage(content="Order B200 was delivered outside the 14-day return window and is not eligible for a refund.", usage_metadata=meta),
        ],
        "scenario_07_policy_qa_citations": [
            AIMessage(content="", tool_calls=[_tool_call("search_policies", {"query": "return window refund timing"})], usage_metadata=meta),
            AIMessage(content="Returns are accepted within 14 days of delivery (returns.md › Return window). Refunds take 5-7 business days to appear (returns.md › Refund timing).", usage_metadata=meta),
        ],
        "scenario_08_uncovered_policy_fallback": [
            AIMessage(content="", tool_calls=[_tool_call("search_policies", {"query": "price matching"})], usage_metadata=meta),
            AIMessage(content="I don't know based on the available policies.", usage_metadata=meta),
        ],
        "scenario_09_refund_approval_over_100": [
            AIMessage(content="", tool_calls=[_tool_call("request_refund", {"order_id": "B201"})], usage_metadata=meta),
            AIMessage(content="Refund request for order B201 ($140.00) exceeds $100 and requires manager approval. Approval request submitted.", usage_metadata=meta),
        ],
    }

    responses = responses_map.get(
        scenario_id,
        [AIMessage(content="Default response.")],
    )
    return ScriptedChatModel(responses=responses)


class ConversationEvaluator:
    """Evaluates multi-turn conversation scenarios against expected outcomes."""

    def __init__(self, live: bool = False) -> None:
        self.live = live

    def evaluate_turn(
        self,
        turn_index: int,
        turn: Turn,
        response_text: str,
        duration_ms: float,
        tokens: dict[str, int],
    ) -> TurnResult:
        """Verify that a single turn response meets keyword, refusal, and outcome criteria."""
        passed = True
        failure_reasons: list[str] = []
        lowered_resp = response_text.lower()

        # Check expected keywords (at least one must match if provided)
        if turn.expected_keywords:
            matched = any(kw.lower() in lowered_resp for kw in turn.expected_keywords)
            if not matched:
                passed = False
                failure_reasons.append(
                    f"Missing expected keywords: {turn.expected_keywords}"
                )

        # Check forbidden keywords (none may match)
        for forbidden in turn.forbidden_keywords:
            if forbidden.lower() in lowered_resp:
                passed = False
                failure_reasons.append(f"Found forbidden keyword: '{forbidden}'")

        # Refusal check
        if turn.is_refusal:
            refusal_markers = [
                "cannot",
                "can't",
                "no order",
                "not found",
                "could not find",
                "outside",
                "not eligible",
                "i don't know",
                "only assist",
            ]
            if not any(marker in lowered_resp for marker in refusal_markers):
                passed = False
                failure_reasons.append("Expected a refusal response but received confirmation.")

        # Approval check
        if turn.is_approval:
            approval_markers = ["approval", "exceeds", "manager", "pending"]
            if not any(marker in lowered_resp for marker in approval_markers):
                passed = False
                failure_reasons.append("Expected an approval notice for order > $100.")

        return TurnResult(
            turn_index=turn_index,
            user_message=turn.message,
            agent_response=response_text,
            passed=passed,
            failure_reasons=failure_reasons,
            duration_ms=duration_ms,
            tokens=tokens,
        )

    def run_scenario(self, scenario: ConversationScenario) -> ScenarioResult:
        """Run all turns in a scenario and return aggregated results."""
        if self.live:
            from app.chat.service import get_chat_service

            chat_service = get_chat_service()
        else:
            mock_model = _build_scripted_model_for_scenario(scenario.scenario_id)
            agent = build_agent(llm=mock_model, checkpointer=MemorySaver())
            chat_service = ChatService(agent=agent)

        session_id = f"eval_{scenario.scenario_id}_{int(time.time())}"
        turn_results: list[TurnResult] = []
        scenario_passed = True
        total_duration_ms = 0.0
        total_tokens = 0
        failure_summary: list[str] = []

        for idx, turn in enumerate(scenario.turns, start=1):
            t_start = time.perf_counter()
            response = chat_service.chat(
                session_id=session_id,
                customer_id=scenario.customer_id,
                message=turn.message,
            )
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            total_duration_ms += elapsed_ms

            tokens = chat_service.get_last_token_usage()
            total_tokens += tokens.get("total_tokens", 0)

            turn_res = self.evaluate_turn(
                turn_index=idx,
                turn=turn,
                response_text=response,
                duration_ms=elapsed_ms,
                tokens=tokens,
            )
            turn_results.append(turn_res)

            if not turn_res.passed:
                scenario_passed = False
                for reason in turn_res.failure_reasons:
                    failure_summary.append(f"Turn {idx}: {reason}")

        return ScenarioResult(
            scenario=scenario,
            passed=scenario_passed,
            turn_results=turn_results,
            total_duration_ms=round(total_duration_ms, 2),
            total_tokens=total_tokens,
            failure_summary=failure_summary,
        )

    def run_all(self, scenarios: list[ConversationScenario] | None = None) -> list[ScenarioResult]:
        """Execute all conversation scenarios."""
        target_scenarios = scenarios or SCENARIOS
        return [self.run_scenario(scenario) for scenario in target_scenarios]
