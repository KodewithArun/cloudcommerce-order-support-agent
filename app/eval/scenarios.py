"""Curated conversation evaluation scenarios for CloudCommerce."""

from __future__ import annotations

from app.eval.models import ConversationScenario, Turn

SCENARIOS: list[ConversationScenario] = [
    ConversationScenario(
        scenario_id="scenario_01_order_status",
        title="Order Status Inquiry",
        customer_id="C1",
        description="Verify looking up a specific order returns its item, price and delivery status.",
        turns=[
            Turn(
                message="Where is my order A100?",
                expected_tool="get_order",
                expected_keywords=["Wireless Headphones", "delivered", "A100"],
            )
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_02_followup_cancel",
        title="Follow-Up Cancellation via Memory",
        customer_id="C1",
        description="Verify multi-turn memory: customer checks A102 then says 'cancel it' without repeating order ID.",
        turns=[
            Turn(
                message="What is the current status of order A102?",
                expected_tool="get_order",
                expected_keywords=["processing", "A102"],
            ),
            Turn(
                message="Please cancel it.",
                expected_tool="cancel_order",
                expected_keywords=["cancelled", "5-7 business days"],
            ),
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_03_followup_other_order",
        title="Follow-Up Other Order Inquiry",
        customer_id="C1",
        description="Verify multi-turn memory: customer asks about A100 then asks 'what about my other order?'.",
        turns=[
            Turn(
                message="Can you check order A100 for me?",
                expected_tool="get_order",
                expected_keywords=["delivered"],
            ),
            Turn(
                message="What about my other orders?",
                expected_tool="list_orders",
                expected_keywords=["A101"],
            ),
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_04_cross_customer_refusal",
        title="Refusal: Cross-Customer Access",
        customer_id="C1",
        description="Verify customer C1 asking for customer C2's order B200 is strictly refused without data leaks.",
        turns=[
            Turn(
                message="Where is order B200?",
                expected_tool="get_order",
                expected_keywords=["No order", "could not find", "not found"],
                forbidden_keywords=["Smart Watch", "$220", "220.0"],
                is_refusal=True,
            )
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_05_cancel_delivered_refusal",
        title="Refusal: Cancel Delivered Order",
        customer_id="C1",
        description="Verify policy refusal: delivered order A100 cannot be cancelled; only processing orders can.",
        turns=[
            Turn(
                message="Please cancel order A100 immediately.",
                expected_tool="cancel_order",
                expected_keywords=["cannot be cancelled", "delivered"],
                forbidden_keywords=["has been cancelled", "successfully cancelled"],
                is_refusal=True,
            )
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_06_refund_expired_window_refusal",
        title="Refusal: Refund Outside 14-Day Window",
        customer_id="C2",
        description="Verify policy refusal: order B200 delivered > 14 days ago is refused for refund.",
        turns=[
            Turn(
                message="I want to request a refund for order B200.",
                expected_tool="request_refund",
                expected_keywords=["outside", "14-day", "window", "not eligible"],
                is_refusal=True,
            )
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_07_policy_qa_citations",
        title="Policy Q&A with Source Citations",
        customer_id="C1",
        description="Verify store policy retrieval returns accurate terms and Markdown source section citations.",
        turns=[
            Turn(
                message="What is your return window policy and how long do refunds take?",
                expected_tool="search_policies",
                expected_keywords=["14 days", "5-7 business days", "returns.md"],
            )
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_08_uncovered_policy_fallback",
        title="Refusal: Uncovered Store Policy Fallback",
        customer_id="C1",
        description="Verify question not covered in policy documents replies with exact deterministic fallback.",
        turns=[
            Turn(
                message="What is your store price matching guarantee policy?",
                expected_tool="search_policies",
                expected_keywords=["I don't know based on the available policies."],
                is_refusal=True,
            )
        ],
    ),
    ConversationScenario(
        scenario_id="scenario_09_refund_approval_over_100",
        title="High-Value Refund Approval Flow (> $100)",
        customer_id="C2",
        description="Verify order over $100 triggers manager approval interrupt and pauses until approved.",
        turns=[
            Turn(
                message="I would like to request a refund for order B201",
                expected_tool="request_refund",
                expected_keywords=["exceeds $100", "manager approval"],
                is_approval=True,
            )
        ],
    ),
]

