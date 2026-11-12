"""FastAPI router for chat endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.chat.models import ChatRequest, ChatResponse
from app.chat.service import ChatService, CustomerMismatchError, get_chat_service

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the order support agent",
    status_code=status.HTTP_200_OK,
)
def chat_endpoint(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Send a customer query to the order support agent.

    Maintains session history across turns via LangGraph checkpoint memory.
    Enforces customer data isolation across sessions and order records.
    """
    try:
        response_text = service.chat(
            session_id=request.session_id,
            customer_id=request.customer_id,
            message=request.message,
        )
    except CustomerMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        session_id=request.session_id,
        customer_id=request.customer_id,
        response=response_text,
    )
