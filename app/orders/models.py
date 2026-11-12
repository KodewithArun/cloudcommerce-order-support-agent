"""Order domain models."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class OrderStatus(StrEnum):
    """Lifecycle states for an order."""

    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUND_REQUESTED = "refund_requested"


class Order(BaseModel):
    """A customer order loaded from the mock order store."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    customer_id: str
    item: str
    price: float
    status: OrderStatus
    delivered_date: date | None = None
