"""FastAPI router for manager approval flows."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.approvals.models import ApproveRequest, ApproveResponse, PendingApproval
from app.audit import log_audit_event
from app.chat.service import ChatService, get_chat_service
from app.orders.service import OrderService, get_order_service

router = APIRouter(tags=["approvals"])


@router.post(
    "/approve",
    response_model=ApproveResponse,
    summary="Approve or reject a high-value refund or pending action",
    status_code=status.HTTP_200_OK,
)
def approve_order(
    request: ApproveRequest,
    order_service: Annotated[OrderService, Depends(get_order_service)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> ApproveResponse:
    """Manager approval endpoint for refunds over $100 and gated operations."""
    result = order_service.approve_refund(
        order_id=request.order_id,
        decision=request.decision,
    )

    if not result.ok and "No order" in result.message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.message,
        )
    if not result.ok and "does not require approval" in result.message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.message,
        )

    resumed_response = None
    if request.session_id:
        try:
            resumed_response = chat_service.resume(
                session_id=request.session_id,
                decision=request.decision,
            )
        except Exception as exc:
            resumed_response = f"Order status updated, but session resume encountered: {exc}"

    log_audit_event(
        "approval",
        {
            "order_id": request.order_id,
            "decision": request.decision,
            "session_id": request.session_id,
            "outcome": result.message,
        },
    )

    order = order_service.get_order(request.order_id, customer_id=order_service._repository.get_by_id(request.order_id).customer_id if order_service._repository.get_by_id(request.order_id) else "")
    current_status = order.status.value if order else ("refunded" if request.decision == "approve" else "refund_rejected")

    return ApproveResponse(
        order_id=request.order_id,
        status=current_status,
        message=result.message,
        resumed_response=resumed_response,
    )


@router.get(
    "/approvals/pending",
    response_model=list[PendingApproval],
    summary="List all orders awaiting manager refund approval",
    status_code=status.HTTP_200_OK,
)
def list_pending_approvals(
    order_service: Annotated[OrderService, Depends(get_order_service)],
) -> list[PendingApproval]:
    """Return all orders with status 'refund_pending_approval'."""
    orders = order_service.list_pending_approvals()
    return [
        PendingApproval(
            order_id=order.order_id,
            customer_id=order.customer_id,
            item=order.item,
            price=order.price,
            status=order.status.value,
        )
        for order in orders
    ]

