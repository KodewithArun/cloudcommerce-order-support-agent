"""Small helpers shared by the provider factories."""

from __future__ import annotations


def require_api_key(value: str | None, env_var: str) -> str:
    """Return ``value`` or raise a clear error naming the missing env var."""
    if not value:
        raise ValueError(f"{env_var} is not set; cannot initialise the provider")
    return value
