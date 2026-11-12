"""Offline tests for orders, tools and policy guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.orders import OrderRepository, OrderService, OrderStatus
from app.orders import tools as order_tools

ORDERS_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.json"
ORDER_TOOLS = order_tools.ORDER_TOOLS


def _settings() -> Settings:
    return Settings(
        orders_path=str(ORDERS_FILE),
        reference_date="2026-11-12",
        return_window_days=14,
    )


@pytest.fixture
def service() -> OrderService:
    return OrderService(OrderRepository.from_json(ORDERS_FILE), _settings())


@pytest.fixture
def patched_service(monkeypatch: pytest.MonkeyPatch, service: OrderService) -> OrderService:
    monkeypatch.setattr(order_tools, "get_order_service", lambda: service)
    return service


def _invoke_tool(name: str, args: dict, customer_id: str) -> str:
    selected = next(tool for tool in ORDER_TOOLS if tool.name == name)
    assert selected.func is not None
    return selected.func(customer_id=customer_id, **args)


def test_get_order_returns_own_order(service: OrderService) -> None:
    order = service.get_order("A100", "C1")

    assert order is not None
    assert order.item == "Wireless Headphones"
    assert order.status is OrderStatus.DELIVERED


def test_get_order_hides_another_customers_order(service: OrderService) -> None:
    assert service.get_order("B200", "C1") is None


def test_list_orders_is_customer_scoped(service: OrderService) -> None:
    order_ids = [order.order_id for order in service.list_orders("C1")]

    assert order_ids == ["A100", "A101", "A102"]


def test_cancel_processing_order(service: OrderService) -> None:
    result = service.cancel_order("A102", "C1")

    assert result.ok
    assert "cancelled" in result.message
    assert service.get_order("A102", "C1").status is OrderStatus.CANCELLED


def test_cancel_shipped_order_is_refused(service: OrderService) -> None:
    result = service.cancel_order("A101", "C1")

    assert not result.ok
    assert "shipped" in result.message


def test_cancel_delivered_order_is_refused(service: OrderService) -> None:
    result = service.cancel_order("A100", "C1")

    assert not result.ok
    assert "delivered" in result.message


def test_cancel_another_customers_order_is_refused(service: OrderService) -> None:
    result = service.cancel_order("B200", "C1")

    assert not result.ok
    assert "No order B200" in result.message


def test_request_refund_inside_return_window(service: OrderService) -> None:
    result = service.request_refund("A100", "C1")

    assert result.ok
    assert "5-7 business days" in result.message
    assert service.get_order("A100", "C1").status is OrderStatus.REFUND_REQUESTED


def test_request_refund_outside_return_window_is_refused(service: OrderService) -> None:
    result = service.request_refund("B200", "C2")

    assert not result.ok
    assert "return window" in result.message


def test_request_refund_requires_delivered_order(service: OrderService) -> None:
    result = service.request_refund("A102", "C1")

    assert not result.ok
    assert "processing" in result.message


def test_tool_schemas_hide_customer_id() -> None:
    for tool in ORDER_TOOLS:
        assert "customer_id" not in tool.args


def test_get_order_tool_uses_injected_customer(patched_service: OrderService) -> None:
    result = _invoke_tool("get_order", {"order_id": "A100"}, "C1")

    assert "Wireless Headphones" in result
    assert "customer_id" not in result


def test_get_order_tool_hides_other_customer(patched_service: OrderService) -> None:
    result = _invoke_tool("get_order", {"order_id": "B200"}, "C1")

    assert "No order B200" in result
    assert "Smart Watch" not in result


def test_list_orders_tool_is_customer_scoped(patched_service: OrderService) -> None:
    result = _invoke_tool("list_orders", {}, "C1")

    assert "A100" in result
    assert "B200" not in result


def test_cancel_tool_changes_status(patched_service: OrderService) -> None:
    result = _invoke_tool("cancel_order", {"order_id": "A102"}, "C1")

    assert "cancelled" in result
    assert patched_service.get_order("A102", "C1").status is OrderStatus.CANCELLED


def test_refund_tool_changes_status(patched_service: OrderService) -> None:
    result = _invoke_tool("request_refund", {"order_id": "A100"}, "C1")

    assert "submitted" in result
    assert patched_service.get_order("A100", "C1").status is OrderStatus.REFUND_REQUESTED
