"""State shared across the agent graph."""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation messages plus the trusted customer ID."""

    customer_id: str
