"""Offline tests for the LangGraph order support agent."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.agent import build_agent
from app.config import Settings
from app.orders import OrderRepository, OrderService
from app.orders import tools as order_tools
from app.rag import UNKNOWN_ANSWER, search_policies

ORDERS_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.json"


class ScriptedChatModel(BaseChatModel):
    """Chat model that returns scripted messages in order."""

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self._index]
        if self._index < len(self.responses) - 1:
            self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class FakeRetriever:
    """Retriever stub returning fixed documents."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def invoke(self, query: str) -> list[Document]:
        return self._documents


def _settings() -> Settings:
    return Settings(
        orders_path=str(ORDERS_FILE),
        reference_date="2026-11-12",
        return_window_days=14,
    )


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> OrderService:
    order_service = OrderService(OrderRepository.from_json(ORDERS_FILE), _settings())
    monkeypatch.setattr(order_tools, "get_order_service", lambda: order_service)
    return order_service


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _run(agent, message: str, customer_id: str = "C1") -> list[BaseMessage]:
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
        }
    )
    return result["messages"]


def test_agent_calls_get_order_and_answers(service: OrderService) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "A100"})]),
            AIMessage(content="Your headphones were delivered on 2026-11-01."),
        ]
    )
    agent = build_agent(llm=model, tools=[order_tools.get_order])

    messages = _run(agent, "Where is order A100?")

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert "Wireless Headphones" in tool_messages[0].content
    assert messages[-1].content == "Your headphones were delivered on 2026-11-01."


def test_agent_hides_another_customers_order(service: OrderService) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("get_order", {"order_id": "B200"})]),
            AIMessage(content="I couldn't find that order for your account."),
        ]
    )
    agent = build_agent(llm=model, tools=[order_tools.get_order])

    messages = _run(agent, "Where is order B200?", customer_id="C1")

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert "No order B200" in tool_messages[0].content
    assert "Smart Watch" not in tool_messages[0].content


def test_agent_ignores_model_supplied_customer_id(service: OrderService) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "get_order",
                        {"order_id": "B200", "customer_id": "C2"},
                    )
                ],
            ),
            AIMessage(content="I couldn't find that order for your account."),
        ]
    )
    agent = build_agent(llm=model, tools=[order_tools.get_order])

    messages = _run(agent, "Where is order B200?", customer_id="C1")

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert "No order B200" in tool_messages[0].content
    assert "Smart Watch" not in tool_messages[0].content


def test_agent_cancels_processing_order(service: OrderService) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_tool_call("cancel_order", {"order_id": "A102"})]),
            AIMessage(content="I cancelled order A102."),
        ]
    )
    agent = build_agent(llm=model, tools=[order_tools.cancel_order])

    messages = _run(agent, "Cancel order A102.")

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert "cancelled" in tool_messages[0].content
    assert service.get_order("A102", "C1").status.value == "cancelled"


def test_agent_policy_tool_returns_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    document = Document(
        page_content="Approved refunds take 5-7 business days.",
        metadata={"source": "returns.md", "section": "Refund timing"},
    )
    monkeypatch.setattr(
        "app.rag.tools.get_policy_retriever",
        lambda: FakeRetriever([document]),
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("search_policies", {"query": "How long do refunds take?"})],
            ),
            AIMessage(content="Refunds take 5-7 business days (returns.md › Refund timing)."),
        ]
    )
    agent = build_agent(llm=model, tools=[search_policies])

    messages = _run(agent, "How long do refunds take?")

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert "returns.md › Refund timing" in tool_messages[0].content
    assert "5-7 business days" in tool_messages[0].content


def test_agent_policy_tool_reports_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.rag.tools.get_policy_retriever",
        lambda: FakeRetriever([]),
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("search_policies", {"query": "Student discount?"})],
            ),
            AIMessage(content=UNKNOWN_ANSWER),
        ]
    )
    agent = build_agent(llm=model, tools=[search_policies])

    messages = _run(agent, "Do you offer student discounts?")

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert "No matching policy found" in tool_messages[0].content
    assert messages[-1].content == UNKNOWN_ANSWER
