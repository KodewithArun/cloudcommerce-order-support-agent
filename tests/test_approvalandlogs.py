"""Tests for Feature 7: Human-in-the-loop approvals, audit logging, and orders-service resilience."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent
from app.approvals.models import ApproveRequest
from app.audit import (
    BACKUP_COUNT,
    MAX_BYTES,
    AuditCallbackHandler,
    get_audit_logger,
    log_audit_event,
)
from app.chat import ChatService, get_chat_service
from app.config import Settings
from app.orders import (
    Order,
    OrderRepository,
    OrderService,
    OrderStatus,
    get_order_service,
)
from app.orders import tools as order_tools
from main import app
from tests.test_agent import ScriptedChatModel, _tool_call

ORDERS_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.json"


@pytest.fixture
def clean_orders() -> list[Order]:
    return [
        Order(
            order_id="A100",
            customer_id="C1",
            item="Wireless Headphones",
            price=85.0,
            status=OrderStatus.DELIVERED,
            delivered_date=None,
        ),
        Order(
            order_id="A102",
            customer_id="C1",
            item="Laptop Stand",
            price=45.0,
            status=OrderStatus.PROCESSING,
            delivered_date=None,
        ),
        Order(
            order_id="B200",
            customer_id="C2",
            item="Smart Watch",
            price=220.0,
            status=OrderStatus.DELIVERED,
            delivered_date=None,
        ),
    ]


@pytest.fixture
def order_service(clean_orders: list[Order], monkeypatch: pytest.MonkeyPatch) -> OrderService:
    settings = Settings(
        orders_path=str(ORDERS_FILE),
        reference_date="2026-11-12",
        return_window_days=14,
    )
    repo = OrderRepository(clean_orders)
    service = OrderService(repo, settings)
    monkeypatch.setattr(order_tools, "get_order_service", lambda: service)
    return service


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Audit Logger & Rotation Tests
# ---------------------------------------------------------------------------


def test_audit_logger_rotation_and_limits() -> None:
    """Audit logger must enforce max 5MB size and rotate with 5 backups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_audit.log"
        logger = get_audit_logger(log_file)

        # Check handler configuration
        handler = logger.handlers[0]
        assert handler.maxBytes == MAX_BYTES
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == BACKUP_COUNT
        assert handler.backupCount == 5

        # Write enough events to trigger a file rotation
        chunk = "X" * 1024  # 1 KB
        for i in range(5200):  # > 5 MB
            log_audit_event("chunk_test", {"index": i, "payload": chunk}, log_file=log_file)

        # Check that rotated backup file was created
        backup_file = Path(f"{log_file}.1")
        assert backup_file.exists()
        assert backup_file.stat().st_size > 0


def test_audit_callback_records_tool_latency_and_args() -> None:
    """Audit callback records tool name, args, duration in ms, and success outcome."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "callback_test.log"
        handler = AuditCallbackHandler(session_id="s123", customer_id="C1", log_file=log_file)

        from uuid import uuid4

        run_id = uuid4()
        handler.on_tool_start(
            serialized={"name": "get_order"},
            input_str="A100",
            run_id=run_id,
            inputs={"order_id": "A100"},
        )
        handler.on_tool_end(output='{"item": "Laptop"}', run_id=run_id)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        call_event = json.loads(lines[0])
        assert call_event["event"] == "tool_call"
        assert call_event["tool"] == "get_order"
        assert call_event["arguments"] == {"order_id": "A100"}
        assert call_event["session_id"] == "s123"
        assert call_event["status"] == "success"
        assert call_event["response_time_ms"] is not None
        assert call_event["response_time_ms"] >= 0.0


def test_complete_generation_tokens_tracking() -> None:
    """Verifies tokens consumed across all model steps in a turn are aggregated."""
    handler = AuditCallbackHandler(session_id="s_tokens", customer_id="C1")

    from uuid import uuid4

    from langchain_core.outputs import ChatGeneration, LLMResult

    # Step 1: Model emits tool call (50 prompt, 15 completion)
    msg1 = AIMessage(content="", usage_metadata={"input_tokens": 50, "output_tokens": 15, "total_tokens": 65})
    res1 = LLMResult(generations=[[ChatGeneration(message=msg1)]])
    run1 = uuid4()
    handler.on_llm_start({}, ["prompt 1"], run_id=run1)
    handler.on_llm_end(res1, run_id=run1)

    # Step 2: Model finishes generation (80 prompt, 25 completion)
    msg2 = AIMessage(
        content="Final reply",
        usage_metadata={"input_tokens": 80, "output_tokens": 25, "total_tokens": 105},
    )
    res2 = LLMResult(generations=[[ChatGeneration(message=msg2)]])
    run2 = uuid4()
    handler.on_llm_start({}, ["prompt 2"], run_id=run2)
    handler.on_llm_end(res2, run_id=run2)

    usage = handler.get_token_usage()
    assert usage["prompt_tokens"] == 130
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 170


# ---------------------------------------------------------------------------
# 2. Human-in-the-loop & /approve Flow Tests
# ---------------------------------------------------------------------------


def test_refund_over_100_triggers_approval_and_resumes(
    client: TestClient, order_service: OrderService
) -> None:
    """Refunds over $100 require manager approval and pause via interrupt."""
    # Order B200 is $220.0 (over $100)
    # Give B200 a delivery date within return window
    b200 = order_service.get_order("B200", "C2")
    assert b200 is not None
    order_service._repository._orders["B200"] = b200.model_copy(
        update={"delivered_date": order_service._today()}
    )

    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("request_refund", {"order_id": "B200"})]),
            AIMessage(content="The refund has been approved by management and completed."),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_order_service] = lambda: order_service

    try:
        # Turn 1: Customer requests refund for $220 order B200
        res = client.post(
            "/chat",
            json={
                "session_id": "sess-b200-approval",
                "customer_id": "C2",
                "message": "I would like a refund for order B200",
            },
        )
        assert res.status_code == 200
        reply = res.json()["response"]
        assert "exceeds $100" in reply
        assert "manager approval" in reply

        # Check pending approvals endpoint
        pending_res = client.get("/approvals/pending")
        assert pending_res.status_code == 200
        pending_orders = pending_res.json()
        assert any(o["order_id"] == "B200" for o in pending_orders)

        # Manager approves the refund via POST /approve
        approve_res = client.post(
            "/approve",
            json={
                "order_id": "B200",
                "decision": "approve",
                "session_id": "sess-b200-approval",
            },
        )
        assert approve_res.status_code == 200
        data = approve_res.json()
        assert data["order_id"] == "B200"
        assert data["status"] == "refunded"
        assert "approved" in data["message"]

        # Verify order status in repository is now REFUNDED
        updated = order_service._repository.get_by_id("B200")
        assert updated is not None
        assert updated.status == OrderStatus.REFUNDED
    finally:
        app.dependency_overrides.clear()


def test_reject_refund_via_approve_endpoint(order_service: OrderService, client: TestClient) -> None:
    """Manager can reject a pending refund request."""
    # Put B200 in pending approval state
    order_service._repository.set_status("B200", OrderStatus.REFUND_PENDING_APPROVAL)

    app.dependency_overrides[get_order_service] = lambda: order_service
    try:
        res = client.post(
            "/approve",
            json=ApproveRequest(order_id="B200", decision="reject").model_dump(),
        )
        assert res.status_code == 200
        assert res.json()["status"] == "refund_rejected"
        assert "rejected" in res.json()["message"]

        order = order_service._repository.get_by_id("B200")
        assert order is not None
        assert order.status == OrderStatus.REFUND_REJECTED
    finally:
        app.dependency_overrides.clear()


def test_approve_nonexistent_order_returns_404(client: TestClient) -> None:
    """Approving an unknown order returns 404."""
    res = client.post(
        "/approve",
        json={"order_id": "NONEXISTENT", "decision": "approve"},
    )
    assert res.status_code == 404


def test_approve_order_not_needing_approval_returns_400(
    client: TestClient, order_service: OrderService
) -> None:
    """Approving an order that doesn't have pending refund returns 400."""
    app.dependency_overrides[get_order_service] = lambda: order_service
    try:
        # A102 is processing, not pending approval
        res = client.post(
            "/approve",
            json={"order_id": "A102", "decision": "approve"},
        )
        assert res.status_code == 400
        assert "does not require approval" in res.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 3. Graceful Orders-Service Failure Handling Tests
# ---------------------------------------------------------------------------


def test_graceful_orders_service_outage_handling(
    client: TestClient, order_service: OrderService
) -> None:
    """When orders service is down, the agent informs the customer without crashing."""
    # Simulate service failure
    order_service.simulate_failure = True

    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A100"})]),
            AIMessage(
                content="Our order system is temporarily experiencing technical difficulties. Please check back in a few moments."
            ),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_order_service] = lambda: order_service

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": "sess-outage-1",
                "customer_id": "C1",
                "message": "Where is order A100?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "technical difficulties" in data["response"] or "temporarily" in data["response"]
    finally:
        app.dependency_overrides.clear()
