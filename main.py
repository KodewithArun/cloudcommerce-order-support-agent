"""FastAPI application for CloudCommerce Order Support."""

from fastapi import FastAPI

from app.chat import chat_router

app = FastAPI(title="CloudCommerce Order Support Agent")
app.include_router(chat_router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Say hello to Cloud Sathi."""
    return {"message": "hello cloud sathi"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
