"""Pydantic schemas for the chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message request."""

    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation session or thread identifier.",
        examples=["session-101"],
    )
    customer_id: str = Field(
        ...,
        min_length=1,
        description="Trusted customer identifier for order data isolation.",
        examples=["C1"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="User's question, follow-up, or request.",
        examples=["Where is my order?"],
    )


class TokenUsage(BaseModel):
    """Token consumption metrics for a model interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    session_id: str = Field(..., description="The conversation session identifier.")
    customer_id: str = Field(..., description="The customer identifier.")
    response: str = Field(..., description="The agent's reply to the user.")
    tokens: TokenUsage | None = Field(
        default=None,
        description="Total tokens consumed across all model generation steps in this turn.",
    )

