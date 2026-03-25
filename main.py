from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import build_agent
from database import connect_db, close_db, connect_sync_db, close_sync_db, get_db, get_sync_collection
from models.stock import StockCreate, StockUpdate, StockResponse
from services.stock_service import StockService
from services.chat_service import ChatService
from services.admin_service import AdminService, lookup_ip_locations
from services import rag_service
from models.chat import ChatRequest, ChatResponse

# ── Shared app state ──────────────────────────────────────────────────────────
app_state: dict = {}


# ── Lifespan: connect DB + build agent on startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    connect_sync_db()
    try:
        app_state["agent"] = build_agent(get_sync_collection("stocks"))
        print("✅ LlamaIndex agent ready")
    except ValueError as e:
        print(f"⚠️  Agent not started: {e}")
        app_state["agent"] = None
    yield
    close_sync_db()
    await close_db()
    app_state.clear()


app = FastAPI(
    title="Stocksight API",
    description="Stock market API powered by FastAPI + MongoDB + LlamaIndex",
    version="3.0.0",
    lifespan=lifespan,
)

import os
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:4200").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency helpers ────────────────────────────────────────────────────────
def get_service() -> StockService:
    return StockService(get_db())

def get_chat_service() -> ChatService:
    return ChatService(get_db())

def get_admin_service() -> AdminService:
    return AdminService(get_db())

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


# ── General ───────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Welcome to Stocksight API", "version": "3.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ── Stock CRUD ────────────────────────────────────────────────────────────────
@app.get("/stocks", response_model=list[StockResponse])
async def get_all_stocks(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Max records to return"),
):
    """Get all stocks with optional pagination."""
    return await get_service().get_all(skip=skip, limit=limit)


@app.get("/stocks/search", response_model=list[StockResponse])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search by symbol, name or sector"),
):
    """Case-insensitive search across symbol, name, and sector."""
    return await get_service().search(q)


@app.get("/stocks/{symbol}", response_model=StockResponse)
async def get_stock(symbol: str):
    """Get a specific stock by ticker symbol."""
    stock = await get_service().get_by_symbol(symbol)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol.upper()}' not found")
    return stock


@app.post("/stocks", response_model=StockResponse, status_code=201)
async def create_stock(stock: StockCreate):
    """Create a new stock entry."""
    existing = await get_service().get_by_symbol(stock.symbol)
    if existing:
        raise HTTPException(
            status_code=409, detail=f"Stock '{stock.symbol.upper()}' already exists"
        )
    return await get_service().create(stock)


@app.put("/stocks/{symbol}", response_model=StockResponse)
async def update_stock(symbol: str, payload: StockUpdate):
    """Partial update — only the fields you provide are changed."""
    updated = await get_service().update(symbol, payload)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol.upper()}' not found")
    return updated


@app.delete("/stocks/{symbol}", response_model=StockResponse)
async def delete_stock(symbol: str):
    """Permanently delete a stock by symbol."""
    deleted = await get_service().delete(symbol)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol.upper()}' not found")
    return deleted


# ── Market analytics ──────────────────────────────────────────────────────────
@app.get("/market/summary")
async def market_summary():
    """High-level summary: total stocks, gainers, losers, flat."""
    return await get_service().market_summary()


@app.get("/market/top-gainer", response_model=StockResponse)
async def top_gainer():
    """Stock with the highest percentage gain today."""
    stock = await get_service().get_top_gainer()
    if not stock:
        raise HTTPException(status_code=404, detail="No stocks available")
    return stock


@app.get("/market/top-loser", response_model=StockResponse)
async def top_loser():
    """Stock with the biggest percentage loss today."""
    stock = await get_service().get_top_loser()
    if not stock:
        raise HTTPException(status_code=404, detail="No stocks available")
    return stock


# ── Agent ─────────────────────────────────────────────────────────────────────
class AgentQuery(BaseModel):
    question: str


@app.post("/agent/query")
async def agent_query(body: AgentQuery):
    """
    Ask the LlamaIndex agent a natural language question about stocks.

    Examples:
      - "What is the price of AAPL?"
      - "Which stock is the top gainer today?"
      - "Give me a market summary"
    """
    agent = app_state.get("agent")
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent unavailable. Check OPENAI_API_KEY is set in your .env file.",
        )
    response = await agent.achat(body.question)
    return {"question": body.question, "answer": str(response)}


# ── Chat sessions ─────────────────────────────────────────────────────────────

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(body: ChatRequest, request: Request):
    """
    Send a question, get a RAG-grounded answer, and persist the exchange.
    Pass session_id to continue an existing session; omit to start a new one.
    """
    ip = get_client_ip(request)
    svc = get_chat_service()

    # Create session on first message; verify it exists if one is provided
    if not body.session_id:
        session_id = await svc.create_session(ip, body.question)
    else:
        existing = await svc.get_session(body.session_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")
        session_id = body.session_id

    # Fetch conversation history for this session
    history = []
    if session_id:
        session = await svc.get_session(session_id)
        if session:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in session.get("messages", [])
            ]

    # Get RAG answer (runs sync in thread pool)
    import asyncio
    try:
        result = await asyncio.to_thread(rag_service.query, body.question, 20, history)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    answer = result["answer"]
    sources = result.get("sources", [])
    metrics = result.get("metrics")

    # Persist both turns
    await svc.append_messages(session_id, body.question, answer, metrics)

    return ChatResponse(session_id=session_id, answer=answer, sources=sources, metrics=metrics)


@app.get("/chat/sessions")
async def list_sessions(request: Request):
    """Return all chat sessions for the requesting IP, newest first."""
    ip = get_client_ip(request)
    return await get_chat_service().get_sessions_by_ip(ip)


@app.get("/chat/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """Return a full session with all messages."""
    session = await get_chat_service().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.patch("/chat/sessions/{session_id}/archive")
async def archive_session(session_id: str):
    """Soft-delete a session — hides it from the UI without removing data."""
    archived = await get_chat_service().archive_session(session_id)
    if not archived:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"archived": session_id}


@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Hard-delete a chat session."""
    deleted = await get_chat_service().delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.get("/admin/stats")
async def admin_stats():
    """Overall session stats for the admin dashboard header."""
    return await get_admin_service().get_stats()


@app.get("/admin/sessions")
async def admin_list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("last_active", pattern="^(session_count|first_seen|last_active)$"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
):
    """Paginated list of all unique IPs with their session counts and activity."""
    return await get_admin_service().get_ips_paginated(page, page_size, sort_by, sort_dir)


@app.post("/admin/ip-locations")
async def admin_ip_locations(ips: list[str]):
    """Batch-resolve IP addresses to city/country using ip-api.com."""
    return await lookup_ip_locations(ips)


@app.get("/admin/sessions/by-ip")
async def admin_sessions_by_ip(
    ip: str = Query(..., description="IP address to filter by"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
):
    """Paginated sessions for a specific IP address."""
    return await get_admin_service().get_sessions_for_ip(ip, page, page_size)


# ── RAG ───────────────────────────────────────────────────────────────────────
class RagQuery(BaseModel):
    question: str
    top_k: int = 20


@app.get("/rag/context")
async def rag_context(
    q: str = Query(..., description="Question to embed and retrieve context for"),
    top_k: int = Query(20, ge=1, le=50),
):
    """
    Fetch raw Pinecone chunks for a question without calling OpenAI.
    Use this to inspect retrieved context and debug relevance.
    """
    import asyncio
    try:
        return await asyncio.to_thread(rag_service.fetch_context, q, top_k)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/rag/query")
async def rag_query(body: RagQuery):
    """
    Answer a question using SEC filing context retrieved from Pinecone.

    Examples:
      - "What were NVIDIA's revenue figures in their latest 10-K?"
      - "What risk factors did Apple mention in their SEC filings?"
    """
    try:
        import asyncio
        result = await asyncio.to_thread(rag_service.query, body.question, body.top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

