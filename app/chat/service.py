"""Chat service coordinating session memory and LangGraph agent execution."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent.graph import get_agent
from app.audit import AuditCallbackHandler


class CustomerMismatchError(Exception):
    """Raised when a request attempts to access a session owned by another customer."""


def extract_message_text(message: BaseMessage) -> str:
    """Safely extract plain text from an AIMessage or BaseMessage.

    Handles string content and structured multimodal/block-based content.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content)


class ChatService:
    """Coordinates conversational order support requests."""

    def __init__(self, agent: CompiledStateGraph | None = None) -> None:
        self._agent = agent
        self.last_token_usage: dict[str, int] = {}

    @property
    def agent(self) -> CompiledStateGraph:
        """Return the compiled LangGraph agent, lazily resolving the shared default."""
        if self._agent is None:
            self._agent = get_agent()
        return self._agent

    def get_last_token_usage(self) -> dict[str, int]:
        """Return the token consumption of the most recent turn."""
        return self.last_token_usage

    def chat(self, *, session_id: str, customer_id: str, message: str) -> str:
        """Run one conversation turn and return the assistant response.

        Args:
            session_id: Unique thread or session identifier.
            customer_id: The authenticated/trusted customer identifier.
            message: The customer's message text.

        Returns:
            The agent's reply text.

        Raises:
            CustomerMismatchError: If the session was already created by another customer.
        """
        turn_start = time.perf_counter()
        handler = AuditCallbackHandler(session_id=session_id, customer_id=customer_id)
        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "callbacks": [handler],
        }

        # Enforce customer data isolation across sessions
        snapshot = self.agent.get_state(config)
        if snapshot.values and "customer_id" in snapshot.values:
            existing_customer = snapshot.values["customer_id"]
            if existing_customer and existing_customer != customer_id:
                raise CustomerMismatchError(
                    f"Session '{session_id}' belongs to another customer."
                )

        result = self.agent.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "customer_id": customer_id,
            },
            config=config,
        )

        self.last_token_usage = handler.get_token_usage()
        from app.audit import log_audit_event

        log_audit_event(
            "chat_turn",
            {
                "session_id": session_id,
                "customer_id": customer_id,
                "duration_ms": round((time.perf_counter() - turn_start) * 1000, 2),
                "tokens": self.last_token_usage,
            },
        )

        # Check for LangGraph dynamic human-in-the-loop interrupts (e.g. refund approval > $100)
        if "__interrupt__" in result and result["__interrupt__"]:
            interrupt_payload = result["__interrupt__"][0].value
            order_id = interrupt_payload.get("order_id", "order")
            amount = interrupt_payload.get("amount", 0.0)
            return (
                f"Refund request of ${amount:.2f} for order {order_id} exceeds $100 "
                "and requires manager approval. Approval request submitted."
            )

        final_message = result["messages"][-1]
        return extract_message_text(final_message)

    def resume(self, *, session_id: str, decision: str) -> str:
        """Resume an interrupted LangGraph session with a human approval decision."""
        from langgraph.types import Command

        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "callbacks": [AuditCallbackHandler(session_id=session_id)],
        }
        result = self.agent.invoke(Command(resume=decision), config=config)
        final_message = result["messages"][-1]
        return extract_message_text(final_message)

    def get_history(self, *, session_id: str, customer_id: str) -> list[BaseMessage]:
        """Return the conversation message history for a customer's session."""
        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        snapshot = self.agent.get_state(config)
        if not snapshot.values:
            return []
        if snapshot.values.get("customer_id") != customer_id:
            raise CustomerMismatchError(
                f"Session '{session_id}' belongs to another customer."
            )
        return snapshot.values.get("messages", [])


_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """Return the shared ChatService singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


def reset_chat_service() -> None:
    """Reset the cached ChatService singleton (for test isolation)."""
    global _chat_service
    _chat_service = None
