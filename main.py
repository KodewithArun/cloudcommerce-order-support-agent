"""Minimal FastAPI application for CloudCommerce."""

from fastapi import FastAPI

app = FastAPI(title="CloudCommerce Order Support Agent")


@app.get("/")
def read_root() -> dict[str, str]:
    """Say hello to Cloud Sathi."""
    return {"message": "hello cloud sathi"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
