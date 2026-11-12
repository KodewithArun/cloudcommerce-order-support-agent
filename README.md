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

## Feature 2: Provider Abstractions

### What I did

- Added a typed `Settings` module using `pydantic-settings` v2
- Built reusable factories around the official LangChain integrations:
  - LLM: Groq, OpenAI, Anthropic
  - Embeddings: Hugging Face (local), OpenAI, fake/offline
- Shared LLM tuning for temperature, retries and timeout
- Clear errors when a provider API key is missing
- Cached settings and provider instances
- Added offline provider tests and an optional live smoke test
- Installed the Hugging Face embedding stack (CPU-only Torch locally)

### Usage

```python
from app.providers import get_chat_model, get_embeddings

answer = get_chat_model().invoke("Where is my order?").content
vector = get_embeddings().embed_query("Where is my order?")
```

### Configure

Switch providers in `.env`:

```bash
LLM_PROVIDER=groq            # groq | openai | anthropic
EMBEDDING_PROVIDER=huggingface  # huggingface | openai | fake
```

### Test

```bash
uv run ruff check .
uv run pytest
RUN_PROVIDER_SMOKE=1 uv run pytest tests/test_provider_smoke.py -s
```

---

## Feature 3: Orders Data, Policy Documents and RAG

### What I did

- Added mock order data in `data/orders.json`
- Added four policy documents in `data/policies/`:
  returns, shipping, warranty and cancellations
- Loaded Markdown with `TextLoader`
- Split by Markdown headers with `MarkdownHeaderTextSplitter`, then by size
  with `RecursiveCharacterTextSplitter`
- Indexed chunks in persistent Chroma at `data/chroma`
- Retrieved with a score-threshold retriever
- Built a retrieve-then-generate chain with `create_stuff_documents_chain`
  and `create_retrieval_chain`
- Added source citations like `returns.md › Return window`
- Returned a deterministic fallback when no policy matches
- Added 9 offline RAG tests and an end-to-end notebook

### Usage

```python
from app.rag import answer_policy_question, format_citations

response = answer_policy_question("How long do refunds take?")
print(response["answer"])
print(format_citations(response["context"]))
```

Uncovered questions return:

```text
I don't know based on the available policies.
```

### Configure

```bash
POLICY_DOCS_PATH=data/policies
RAG_CHUNK_SIZE=700
RAG_CHUNK_OVERLAP=120
RAG_TOP_K=4
RAG_MIN_SCORE=0.35
CHROMA_PERSIST_PATH=data/chroma
CHROMA_COLLECTION_NAME=policies
```

### Test

```bash
uv run pytest tests/test_rag.py -q
```

### Notebook

Open `app/rag/rag_demo.ipynb` to inspect loading, indexing, retrieval,
relevance scores, answers and citations interactively.

---

## Feature 4 + 5: Order Tools and LangGraph Agent

### What I did

- Added typed `Order` and `OrderStatus` models
- Added a JSON-backed, customer-scoped `OrderRepository`
- Kept order changes in memory for the current process
- Added `OrderService` guards:
  - cancel only while status is `processing`
  - refund only for delivered orders inside the 14-day return window
  - cross-customer orders return a generic not-found message
- Added LangChain order tools: `get_order`, `list_orders`, `cancel_order`,
  `request_refund`
- Added `search_policies`, which reuses the RAG retriever and returns excerpts
  with `source › section` citations
- Added an explicit LangGraph agent:
  - `AgentState` with messages and trusted `customer_id`
  - assistant and tool nodes
  - `ToolNode` and `tools_condition` routing
  - system prompt for policy citations, refusals and customer isolation
- Injected `customer_id` from graph state with `InjectedState`, so the model
  cannot choose or override it
- Added offline order, tool and agent tests

### Architecture

```text
START -> assistant -> tools_condition -> tools -> assistant -> END
```

### Usage

```python
from langchain_core.messages import HumanMessage

from app.agent import get_agent

result = get_agent().invoke(
    {
        "messages": [HumanMessage(content="Where is order A100?")],
        "customer_id": "C1",
    },
    config={"configurable": {"thread_id": "session-1"}},
)
print(result["messages"][-1].content)
```

### Configure

```bash
ORDERS_PATH=data/orders.json
RETURN_WINDOW_DAYS=14
REFERENCE_DATE=        # optional ISO date for reproducible tests
```

### Test

```bash
uv run pytest tests/test_orders.py tests/test_agent.py -q
```

Refund approval, audit logging and order-service failures are implemented in
later features.

---

## Feature 6: Conversation Memory and `POST /chat`

### What I did

- Integrated official LangGraph checkpoint persistence using `MemorySaver`
- Configured conversation memory with `{"configurable": {"thread_id": session_id}}`
- Supported multi-turn contextual follow-ups such as "cancel it" and "what about my other order?"
- Added FastAPI `POST /chat` endpoint taking `session_id`, `customer_id`, and `message`
- Enforced session-level customer isolation: requests attempting to access another customer's `session_id` are rejected with HTTP 403 Forbidden
- Added typed `ChatRequest` and `ChatResponse` schemas
- Added `ChatService` with clean message text extraction and dependency injection
- Added 8 offline unit & integration tests covering single-turn, multi-turn memory, follow-up cancellations, session hijacking protection, and policy refusals

### Endpoints

| Method | Path | Request Body | Description |
|--------|------|--------------|-------------|
| POST | `/chat` | `{"session_id": "...", "customer_id": "...", "message": "..."}` | Send message to support agent with session memory |

### Usage

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-101",
    "customer_id": "C1",
    "message": "Where is my order A100?"
  }'
```

Response:

```json
{
  "session_id": "session-101",
  "customer_id": "C1",
  "response": "Your order A100 was delivered on 2026-11-01."
}
```

Follow-up using the same `session_id`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-101",
    "customer_id": "C1",
    "message": "Can you refund it?"
  }'
```

### Test

```bash
uv run pytest tests/test_chat.py -q
```

---

## Feature 7: Approval Flow, Audit Logging and Failure Handling

### What I did

- Integrated official LangGraph Human-in-the-Loop (`interrupt()` and `Command(resume=...)`):
  - Refund requests over $100 automatically pause graph execution via dynamic interrupt and enter `refund_pending_approval` status.
- Added FastAPI approval endpoints:
  - `POST /approve`: Approves or rejects pending high-value refunds and resumes the paused conversation thread.
  - `GET /approvals/pending`: Lists all orders currently awaiting manager approval.
- Implemented professional rotating audit logger (`app/audit/`):
  - Configured `RotatingFileHandler` with `maxBytes=5MB` (5,242,880 bytes) and `backupCount=5` to prevent unbounded log growth.
  - Captures structured JSONL records: timestamp, session ID, customer ID, tool name, arguments, execution duration in milliseconds, outcomes, token usage, and token generation speed (tps).
- Added resilient orders-service failure recovery:
  - Configured `ToolNode(handle_tool_errors=...)` so network/database errors return friendly customer messages rather than crashing with HTTP 500.
- Added 7 comprehensive tests in `tests/test_feature7.py` covering HITL interrupts, `/approve`, log rotation, and failure resilience.

### Endpoints

| Method | Path | Request Body | Description |
|--------|------|--------------|-------------|
| POST | `/approve` | `{"order_id": "...", "decision": "approve", "session_id": "..."}` | Approve or reject a refund over $100 |
| GET | `/approvals/pending` | None | List orders currently awaiting approval |

### Usage

1. Request a refund over $100 (e.g. order `B200` at $220.00):
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-102",
    "customer_id": "C2",
    "message": "I would like a refund for order B200"
  }'
```

Response:
```json
{
  "session_id": "session-102",
  "customer_id": "C2",
  "response": "Refund request of $220.00 for order B200 exceeds $100 and requires manager approval. Approval request submitted."
}
```

2. Manager checks pending approvals:
```bash
curl http://127.0.0.1:8000/approvals/pending
```

3. Manager approves the refund via `POST /approve`:
```bash
curl -X POST http://127.0.0.1:8000/approve \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "B200",
    "decision": "approve",
    "session_id": "session-102"
  }'
```

### Test

```bash
uv run pytest tests/test_approvalandlogs.py -q
```

---

## Feature 8: Conversation Evaluation Suite

### What I did

- Created 9 multi-turn conversation scenarios in `app/eval/scenarios.py` with explicit expected outcomes:
  1. **Order Status Inquiry:** Lookup order A100.
  2. **Follow-Up Cancellation via Memory:** Multi-turn order check followed by "cancel it".
  3. **Follow-Up Other Order Inquiry:** Multi-turn "what about my other order?".
  4. **Refusal 1 — Cross-Customer Access:** Customer C1 asks for C2's order B200 (strictly denied without leaks).
  5. **Refusal 2 — Cancel Delivered Order:** Refuses cancellation of delivered order A100.
  6. **Refusal 3 — Refund Outside 14-Day Window:** Refuses refund for order B200 delivered > 14 days ago.
  7. **Policy Q&A with Source Citations:** Queries return policy and checks for citations (`returns.md › Return window`).
  8. **Refusal 4 — Uncovered Policy Fallback:** Asks about price matching and verifies deterministic fallback: *"I don't know based on the available policies."*.
  9. **High-Value Refund Approval Flow:** Verifies refund over $100 pauses for manager approval.
- Implemented `ConversationEvaluator` (`app/eval/runner.py`) to validate tool calls, keywords, refusals, latencies, and token usage.
- Added executable CLI tool `cloudcommerce-eval` (`app.eval.run_tests:main`) configured via `pyproject.toml` scripts.
- Formats reports as colorized ASCII summary tables with turn-by-turn tracking, latency, tokens, and exit codes for CI.
- Added automated offline pytest test in `tests/test_eval.py`.

### Run

```bash
# Run full evaluation suite
uv run cloudcommerce-eval

# Run with turn-by-turn verbose details
uv run cloudcommerce-eval -v

# Run against live configured LLM provider
uv run cloudcommerce-eval --live
```

### Test

```bash
uv run pytest tests/test_eval.py -q
```

---

## Status

- [x] Feature 1: Project setup and minimal FastAPI service
- [x] Feature 2: Provider abstractions (Groq + Hugging Face + offline fallback)
- [x] Feature 3: Orders data, policy documents and RAG with citations
- [x] Feature 4: Order tools and safety guards
- [x] Feature 5: LangGraph agent
- [x] Feature 6: Conversation memory and `POST /chat`
- [x] Feature 7: Approval flow, audit log and failure handling
- [x] Feature 8: Conversation eval suite
- [ ] Feature 9: Final README with architecture diagram and learnings
