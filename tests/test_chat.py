"""Offline tests for the conversation memory and POST /chat endpoint."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from pydantic import PrivateAttr

from app.agent.graph import build_agent
from app.chat import ChatService, get_chat_service
from app.config import Settings
from app.orders import OrderRepository, OrderService
from app.orders import tools as order_tools
from main import app

ORDERS_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.json"


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class RecordingChatModel(BaseChatModel):
    """Mock chat model that records prompt history and emits scripted responses."""

    responses: list[AIMessage]
    recorded_prompts: list[list[BaseMessage]] = []
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "recording-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RecordingChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recorded_prompts.append(list(messages))
        response = self.responses[self._index]
        if self._index < len(self.responses) - 1:
            self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


@pytest.fixture
def order_service(monkeypatch: pytest.MonkeyPatch) -> OrderService:
    settings = Settings(
        orders_path=str(ORDERS_FILE),
        reference_date="2026-11-12",
        return_window_days=14,
    )
    service = OrderService(OrderRepository.from_json(ORDERS_FILE), settings)
    monkeypatch.setattr(order_tools, "get_order_service", lambda: service)
    return service


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_chat_single_turn(client: TestClient, order_service: OrderService) -> None:
    model = RecordingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A100"})]),
            AIMessage(content="Order A100 was delivered on 2026-11-01."),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": "session-1",
                "customer_id": "C1",
                "message": "Where is my order A100?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session-1"
        assert data["customer_id"] == "C1"
        assert "Order A100 was delivered" in data["response"]
    finally:
        app.dependency_overrides.clear()


def test_chat_conversation_memory_followup_cancel(
    client: TestClient, order_service: OrderService
) -> None:
    """Verifies that follow-up 'cancel it' retains order context from turn 1."""
    # A102 is processing for C1
    model = RecordingChatModel(
        responses=[
            # Turn 1: user asks status of A102
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A102"})]),
            AIMessage(content="Order A102 is currently processing."),
            # Turn 2: user says "cancel it", model infers A102 from memory
            AIMessage(content="", tool_calls=[_tool_call("cancel_order", {"order_id": "A102"})]),
            AIMessage(content="I have cancelled order A102 for you."),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        # Turn 1
        res1 = client.post(
            "/chat",
            json={
                "session_id": "session-mem-1",
                "customer_id": "C1",
                "message": "What is the status of order A102?",
            },
        )
        assert res1.status_code == 200
        assert "A102 is currently processing" in res1.json()["response"]

        # Turn 2
        res2 = client.post(
            "/chat",
            json={
                "session_id": "session-mem-1",
                "customer_id": "C1",
                "message": "Please cancel it.",
            },
        )
        assert res2.status_code == 200
        assert "cancelled order A102" in res2.json()["response"]

        # Verify that turn 2 LLM prompt received prior conversation history
        second_turn_prompt = model.recorded_prompts[-2]  # Prompt before cancel_order tool call
        prompt_texts = [
            m.content for m in second_turn_prompt if isinstance(m.content, str) and m.content
        ]
        assert any("What is the status of order A102?" in text for text in prompt_texts)
        assert any("Please cancel it." in text for text in prompt_texts)

        # Verify order state actually updated in repository
        cancelled_order = order_service.get_order("A102", "C1")
        assert cancelled_order is not None
        assert cancelled_order.status.value == "cancelled"
    finally:
        app.dependency_overrides.clear()


def test_chat_conversation_memory_other_order(
    client: TestClient, order_service: OrderService
) -> None:
    """Verifies follow-up 'what about my other order?' resolves with conversation context."""
    model = RecordingChatModel(
        responses=[
            # Turn 1: user asks about A100
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A100"})]),
            AIMessage(content="Order A100 is delivered."),
            # Turn 2: user asks 'what about my other order?'
            AIMessage(content="", tool_calls=[_tool_call("list_orders", {})]),
            AIMessage(content="Your other order A101 is currently processing."),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        # Turn 1
        res1 = client.post(
            "/chat",
            json={
                "session_id": "session-mem-2",
                "customer_id": "C1",
                "message": "Where is order A100?",
            },
        )
        assert res1.status_code == 200

        # Turn 2
        res2 = client.post(
            "/chat",
            json={
                "session_id": "session-mem-2",
                "customer_id": "C1",
                "message": "What about my other order?",
            },
        )
        assert res2.status_code == 200
        assert "A101 is currently processing" in res2.json()["response"]
    finally:
        app.dependency_overrides.clear()


def test_chat_cross_customer_session_hijack_rejected(client: TestClient) -> None:
    """A customer cannot hijack or read another customer's session thread."""
    model = RecordingChatModel(
        responses=[
            AIMessage(content="Hello customer 1!"),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        # Customer C1 initiates session
        res1 = client.post(
            "/chat",
            json={
                "session_id": "c1-private-session",
                "customer_id": "C1",
                "message": "Hello!",
            },
        )
        assert res1.status_code == 200

        # Customer C2 attempts to access same session ID
        res2 = client.post(
            "/chat",
            json={
                "session_id": "c1-private-session",
                "customer_id": "C2",
                "message": "What orders do I have?",
            },
        )
        assert res2.status_code == 403
        assert "belongs to another customer" in res2.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_chat_cross_customer_order_access_refusal(
    client: TestClient, order_service: OrderService
) -> None:
    """When customer C1 asks for customer C2's order, the tool refuses and LLM replies."""
    model = RecordingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "B200"})]),
            AIMessage(content="I could not find order B200 associated with your account."),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": "session-c1-order-cross",
                "customer_id": "C1",
                "message": "Where is order B200?",
            },
        )
        assert response.status_code == 200
        assert "could not find order B200" in response.json()["response"]
    finally:
        app.dependency_overrides.clear()


def test_chat_refuse_unsupported_cancellation(
    client: TestClient, order_service: OrderService
) -> None:
    """Cancellation of a delivered order is refused per policy."""
    model = RecordingChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("cancel_order", {"order_id": "A100"})]),
            AIMessage(
                content="I cannot cancel order A100 because it has already been delivered. You may request a refund instead."
            ),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": "session-cancel-refusal",
                "customer_id": "C1",
                "message": "Cancel order A100.",
            },
        )
        assert response.status_code == 200
        assert "cannot cancel order A100" in response.json()["response"]
    finally:
        app.dependency_overrides.clear()


def test_chat_refuse_off_topic_query(client: TestClient) -> None:
    """Off-topic queries receive polite domain boundary refusal."""
    model = RecordingChatModel(
        responses=[
            AIMessage(
                content="I can only assist with CloudCommerce orders, shipping, cancellations, refunds, and store policies."
            ),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": "session-off-topic",
                "customer_id": "C1",
                "message": "What is the capital of France?",
            },
        )
        assert response.status_code == 200
        assert "only assist with CloudCommerce" in response.json()["response"]
    finally:
        app.dependency_overrides.clear()


def test_chat_validation_error(client: TestClient) -> None:
    """Empty or missing fields fail with 422 Unprocessable Entity."""
    # Empty message
    res = client.post(
        "/chat",
        json={"session_id": "s1", "customer_id": "C1", "message": ""},
    )
    assert res.status_code == 422

    # Missing session_id
    res = client.post(
        "/chat",
        json={"customer_id": "C1", "message": "hello"},
    )
    assert res.status_code == 422


def test_chat_service_get_history(order_service: OrderService) -> None:
    """Unit test get_history and isolation in ChatService directly."""
    model = RecordingChatModel(
        responses=[
            AIMessage(content="Hi there!"),
        ]
    )
    agent = build_agent(llm=model, checkpointer=MemorySaver())
    chat_service = ChatService(agent=agent)

    chat_service.chat(session_id="hist-1", customer_id="C1", message="Hello")
    history = chat_service.get_history(session_id="hist-1", customer_id="C1")
    assert len(history) == 2
    assert history[0].content == "Hello"
    assert history[1].content == "Hi there!"

    # Customer C2 cannot read history
    with pytest.raises(Exception, match="belongs to another customer"):
        chat_service.get_history(session_id="hist-1", customer_id="C2")
