"""LangChain tools for order support."""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.orders.models import Order
from app.orders.service import get_order_service

CustomerId = Annotated[str, InjectedState("customer_id")]


def _order_payload(order: Order) -> dict[str, Any]:
    """Serialize an order without exposing its customer ID."""
    return order.model_dump(mode="json", exclude={"customer_id"})


@tool
def get_order(order_id: str, customer_id: CustomerId) -> str:
    """Get one order for the current customer by its order ID."""
    order = get_order_service().get_order(order_id, customer_id)
    if order is None:
        return f"No order {order_id} found for this customer."
    return json.dumps(_order_payload(order), ensure_ascii=False)


@tool
def list_orders(customer_id: CustomerId) -> str:
    """List all orders belonging to the current customer."""
    orders = get_order_service().list_orders(customer_id)
    if not orders:
        return "No orders found for this customer."
    return json.dumps([_order_payload(order) for order in orders], ensure_ascii=False)


@tool
def cancel_order(order_id: str, customer_id: CustomerId) -> str:
    """Cancel an order for the current customer.

    Only orders whose status is still processing can be cancelled.
    """
    return get_order_service().cancel_order(order_id, customer_id).message


@tool
def request_refund(order_id: str, customer_id: CustomerId) -> str:
    """Request a refund for a delivered order for the current customer.

    Only delivered orders inside the return window are eligible.
    """
    return get_order_service().request_refund(order_id, customer_id).message


ORDER_TOOLS = [get_order, list_orders, cancel_order, request_refund]
