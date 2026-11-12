"""Order operations with policy checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from app.config import Settings, get_settings
from app.orders.models import Order, OrderStatus
from app.orders.repository import OrderRepository, get_order_repository


@dataclass(frozen=True)
class ActionResult:
    """Outcome of an order-changing operation."""

    ok: bool
    message: str


class OrderService:
    """Customer-scoped order lookups and guarded status changes."""

    def __init__(self, repository: OrderRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def _today(self) -> date:
        if self._settings.reference_date:
            return date.fromisoformat(self._settings.reference_date)
        return date.today()

    @staticmethod
    def _not_found(order_id: str) -> ActionResult:
        return ActionResult(False, f"No order {order_id} found for this customer.")

    def get_order(self, order_id: str, customer_id: str) -> Order | None:
        """Return one of the customer's orders, or ``None``."""
        return self._repository.get(order_id, customer_id)

    def list_orders(self, customer_id: str) -> list[Order]:
        """Return all orders belonging to the customer."""
        return self._repository.list_for(customer_id)

    def cancel_order(self, order_id: str, customer_id: str) -> ActionResult:
        """Cancel an order only while it is still processing."""
        order = self.get_order(order_id, customer_id)
        if order is None:
            return self._not_found(order_id)
        if order.status != OrderStatus.PROCESSING:
            return ActionResult(
                False,
                f"Order {order.order_id} is {order.status.value} and cannot be cancelled.",
            )

        self._repository.set_status(order.order_id, OrderStatus.CANCELLED)
        return ActionResult(
            True,
            f"Order {order.order_id} has been cancelled. Any refund will take 5-7 business days.",
        )

    def request_refund(self, order_id: str, customer_id: str) -> ActionResult:
        """Request a refund for a delivered order inside the return window."""
        order = self.get_order(order_id, customer_id)
        if order is None:
            return self._not_found(order_id)
        if order.status != OrderStatus.DELIVERED:
            return ActionResult(
                False,
                f"Order {order.order_id} is {order.status.value}; "
                "refunds are only available for delivered orders.",
            )
        if order.delivered_date is None:
            return ActionResult(
                False,
                f"Order {order.order_id} has no delivery date and is not eligible for a refund.",
            )

        days_since_delivery = (self._today() - order.delivered_date).days
        if days_since_delivery < 0:
            return ActionResult(
                False,
                f"Order {order.order_id} has a delivery date in the future.",
            )
        if days_since_delivery > self._settings.return_window_days:
            return ActionResult(
                False,
                f"Order {order.order_id} was delivered {days_since_delivery} days ago, "
                f"outside the {self._settings.return_window_days}-day return window.",
            )

        self._repository.set_status(order.order_id, OrderStatus.REFUND_REQUESTED)
        return ActionResult(
            True,
            f"Refund request for order {order.order_id} submitted. "
            "Refunds take 5-7 business days after processing.",
        )


@lru_cache(maxsize=1)
def get_order_service() -> OrderService:
    """Return the shared order service configured from the environment."""
    return OrderService(get_order_repository(), get_settings())
