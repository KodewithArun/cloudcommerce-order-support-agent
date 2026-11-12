"""Customer-scoped order repository backed by the mock JSON data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.orders.models import Order, OrderStatus


class OrderRepository:
    """In-memory order store.

    Customer scoping is enforced here so callers cannot read another
    customer's order by knowing its order ID.
    """

    def __init__(self, orders: Iterable[Order]) -> None:
        self._orders: dict[str, Order] = {}
        for order in orders:
            if order.order_id in self._orders:
                raise ValueError(f"Duplicate order ID: {order.order_id}")
            self._orders[order.order_id] = order

    @classmethod
    def from_json(cls, path: str | Path) -> OrderRepository:
        """Load orders from the JSON fixture at ``path``."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of orders in {path}")
        return cls(Order.model_validate(item) for item in payload)

    def get(self, order_id: str, customer_id: str) -> Order | None:
        """Return an order only when it belongs to ``customer_id``."""
        order = self._orders.get(order_id)
        if order is None or order.customer_id != customer_id:
            return None
        return order

    def list_for(self, customer_id: str) -> list[Order]:
        """Return the customer's orders sorted by order ID."""
        return sorted(
            (order for order in self._orders.values() if order.customer_id == customer_id),
            key=lambda order: order.order_id,
        )

    def get_by_id(self, order_id: str) -> Order | None:
        """Return an order by ID without customer scoping (internal/manager use)."""
        return self._orders.get(order_id)

    def all(self) -> list[Order]:
        """Return all orders across all customers (internal/manager use)."""
        return list(self._orders.values())

    def set_status(self, order_id: str, status: OrderStatus) -> Order:
        """Replace an order's status and return the updated model."""
        updated = self._orders[order_id].model_copy(update={"status": status})
        self._orders[order_id] = updated
        return updated


@lru_cache(maxsize=1)
def get_order_repository() -> OrderRepository:
    """Return the shared order repository configured from the environment."""
    return OrderRepository.from_json(get_settings().orders_path)
