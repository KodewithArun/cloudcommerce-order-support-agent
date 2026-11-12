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

## Status

- [x] Feature 1: Project setup and minimal FastAPI service
- [x] Feature 2: Provider abstractions (Groq + Hugging Face + offline fallback)
- [x] Feature 3: Orders data, policy documents and RAG with citations
- [ ] Feature 4: Order tools and safety guards
- [ ] Feature 5: LangGraph agent
- [ ] Feature 6: Conversation memory and `POST /chat`
- [ ] Feature 7: Approval flow, audit log and failure handling
- [ ] Feature 8: Conversation eval suite
- [ ] Feature 9: Final README with architecture diagram and learnings
