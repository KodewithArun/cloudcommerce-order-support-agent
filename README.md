# CloudCommerce Order Support Agent

## Description

An AI-powered order support agent for CloudCommerce, built with FastAPI and LangGraph. Features tool-based order management, policy-aware RAG with source citations, conversational memory, customer data isolation, and automated conversation testing.

## Requirements to achieve

- LangGraph or Google ADK agent with tools: `get_order`, `list_orders`,
  `cancel_order`, `request_refund`
- RAG over policy documents with source citations; say "I don't know" when the
  policies do not cover a question
- Conversation memory for follow-ups such as "cancel it" and "what about my
  other order?"
- FastAPI `POST /chat` accepting `session_id`, `customer_id`, `message`
- Refuse cross-customer order access
- Refuse cancellations/refunds that the policy does not allow
- At least 8 test conversations with expected outcomes, including 2+ refusals,
  plus a runner that prints pass/fail
- README with setup, architecture diagram, model/RAG choices and learnings

Bonus: `/approve` for refunds over $100, tool-call logging with response time,
and graceful handling of orders-service failures.

---

## Feature 1: Project Setup & Minimal FastAPI Service

### What I did

- Initialized the Python project with `uv` and Python 3.12
- Added runtime dependencies: FastAPI, Uvicorn, Pydantic Settings, LangGraph,
  LangChain Core, Groq, Hugging Face, NumPy, HTTPX
- Added dev dependencies: pytest, pytest-asyncio, ruff
- Added `.env.example` for provider, RAG, business and storage settings
- Added `.gitignore` for secrets, caches and runtime files
- Created `main.py` with a hello endpoint and a health check
- Added `tests/test_main.py` with 2 passing tests
- Added a local gitignored `.agent-context.md` for agent context
- Verified `ruff` and `pytest` pass

### Endpoints

| Method | Path | Response |
|--------|------|----------|
| GET | `/` | `{"message": "hello cloud sathi"}` |
| GET | `/health` | `{"status": "ok"}` |

### Run

```bash
uv sync
cp .env.example .env
uv run uvicorn main:app --reload
```

OpenAPI docs: `http://127.0.0.1:8000/docs`

### Test

```bash
uv run ruff check .
uv run pytest
```

---

## Status

- [x] Feature 1: Project setup and minimal FastAPI service
- [ ] Feature 2: Provider abstractions (Groq + Hugging Face + offline fallback)
- [ ] Feature 3: Orders data, policy documents and RAG with citations
- [ ] Feature 4: Order tools and safety guards
- [ ] Feature 5: LangGraph agent
- [ ] Feature 6: Conversation memory and `POST /chat`
- [ ] Feature 7: Approval flow, audit log and failure handling
- [ ] Feature 8: Conversation eval suite
- [ ] Feature 9: Final README with architecture diagram and learnings
