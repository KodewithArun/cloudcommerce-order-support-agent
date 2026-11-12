"""Pydantic schemas for the human-in-the-loop approval flow."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApproveRequest(BaseModel):
    """Payload to approve or reject a pending order action."""

    order_id: str = Field(..., min_length=1, description="Order identifier to approve or reject.")
    decision: str = Field(
        default="approve",
        description="Approval decision: 'approve' or 'reject'.",
        examples=["approve", "reject"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional chat session thread ID to resume after approval.",
    )


class ApproveResponse(BaseModel):
    """Result of the approval action."""

    order_id: str
    status: str
    message: str
    resumed_response: str | None = None


class PendingApproval(BaseModel):
    """Summary of an order awaiting manager approval."""

    order_id: str
    customer_id: str
    item: str
    price: float
    status: str

