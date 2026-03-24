# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env          # Windows
cp .env.example .env            # Linux/macOS
# Then fill in OPENAI_API_KEY and MONGODB_URL in .env
```

## Running

```bash
python main.py
# or
uvicorn main:app --reload
```

API available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MONGODB_URL` | Yes | `mongodb://localhost:27017` | MongoDB connection string |
| `DATABASE_NAME` | No | `stocksightDB` | Database name |
| `OPENAI_API_KEY` | Yes (for agent) | — | OpenAI key for LlamaIndex agent |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model for agent |

If `OPENAI_API_KEY` is missing, the app still starts but `/agent/query` returns 503.

## Architecture

The app has three layers:

**FastAPI (`main.py`)** — HTTP entry point. Manages lifespan (DB connections + agent init). Routes delegate to `StockService` for all DB operations. The agent is stored in `app_state` dict and accessed via `/agent/query`.

**Service layer (`services/stock_service.py`)** — All async MongoDB operations go through `StockService`. Uses Motor (async pymongo) for FastAPI routes. Symbol lookups are always uppercased. The `_serialize()` helper converts `_id` ObjectId to string `id`.

**LlamaIndex agent (`agent.py`)** — A separate synchronous pymongo client is used exclusively by the agent tools (Motor's async client cannot be used in LlamaIndex's sync tool loop). `build_agent()` wires five `FunctionTool`s against the sync collection and returns an `OpenAIAgent`.

**Database (`database.py`)** — Maintains two separate MongoDB clients: async (`Motor`) for FastAPI routes, sync (`pymongo`) for the agent. Both connect to the same `stocks` collection. A unique index on `symbol` is enforced on startup.

**Models (`models/stock.py`)** — Pydantic v2 models. `StockCreate` (POST body) and `StockUpdate` (PUT body, all fields optional) extend `StockBase`. `StockResponse` adds `id`, `created_at`, `updated_at`.

## Key Design Notes

- Ticker symbols are always stored and queried as uppercase.
- `PUT /stocks/{symbol}` is a partial update — only non-null fields are written.
- The `/stocks/search` route must be declared **before** `/stocks/{symbol}` in `main.py` to avoid FastAPI treating `search` as a symbol parameter.
- The agent tools are thin wrappers that close over the sync `Collection` via lambdas — they do not go through `StockService`.
