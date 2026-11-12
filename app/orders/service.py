"""Order operations with policy checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from app.config import Settings, get_settings
from app.orders.models import Order, OrderStatus
from app.orders.repository import OrderRepository, get_order_repository


class OrderServiceError(Exception):
    """Base exception for order service operations."""


class OrderServiceUnavailableError(OrderServiceError):
    """Raised when the orders service or database is unavailable."""


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
        self.simulate_failure: bool = False

    def _check_availability(self) -> None:
        if self.simulate_failure:
            raise OrderServiceUnavailableError("Orders service is temporarily unavailable.")

    def _today(self) -> date:
        if self._settings.reference_date:
            return date.fromisoformat(self._settings.reference_date)
        return date.today()

    @staticmethod
    def _not_found(order_id: str) -> ActionResult:
        return ActionResult(False, f"No order {order_id} found for this customer.")

    def get_order(self, order_id: str, customer_id: str) -> Order | None:
        """Return one of the customer's orders, or ``None``."""
        self._check_availability()
        return self._repository.get(order_id, customer_id)

    def list_orders(self, customer_id: str) -> list[Order]:
        """Return all orders belonging to the customer."""
        self._check_availability()
        return self._repository.list_for(customer_id)

    def cancel_order(self, order_id: str, customer_id: str) -> ActionResult:
        """Cancel an order only while it is still processing."""
        self._check_availability()
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
        self._check_availability()
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

        if order.price > 100.0:
            self._repository.set_status(order.order_id, OrderStatus.REFUND_PENDING_APPROVAL)
            return ActionResult(
                True,
                f"Refund request for order {order.order_id} (${order.price:.2f}) exceeds $100 "
                "and requires manager approval. Approval request submitted.",
            )

        self._repository.set_status(order.order_id, OrderStatus.REFUND_REQUESTED)
        return ActionResult(
            True,
            f"Refund request for order {order.order_id} submitted. "
            "Refunds take 5-7 business days after processing.",
        )

    def approve_refund(self, order_id: str, decision: str = "approve") -> ActionResult:
        """Approve or reject a refund request (for store managers)."""
        self._check_availability()
        order = self._repository.get_by_id(order_id)
        if order is None:
            return ActionResult(False, f"No order {order_id} found.")
        if order.status not in (OrderStatus.REFUND_PENDING_APPROVAL, OrderStatus.REFUND_REQUESTED):
            return ActionResult(
                False,
                f"Order {order.order_id} has status '{order.status.value}' and does not require approval.",
            )

        if decision.lower() in ("approve", "approved"):
            self._repository.set_status(order.order_id, OrderStatus.REFUNDED)
            return ActionResult(
                True,
                f"Refund of ${order.price:.2f} for order {order.order_id} has been approved and processed.",
            )

        self._repository.set_status(order.order_id, OrderStatus.REFUND_REJECTED)
        return ActionResult(
            False,
            f"Refund request for order {order.order_id} was rejected by management.",
        )

    def list_pending_approvals(self) -> list[Order]:
        """List all orders currently awaiting refund approval."""
        self._check_availability()
        return [
            order
            for order in self._repository.all()
            if order.status == OrderStatus.REFUND_PENDING_APPROVAL
        ]


@lru_cache(maxsize=1)
def get_order_service() -> OrderService:
    """Return the shared order service configured from the environment."""
    return OrderService(get_order_repository(), get_settings())
