from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_says_hello_cloud_sathi() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "hello cloud sathi"}


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
