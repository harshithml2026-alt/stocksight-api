from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agent import build_agent

# Agent is stored here after startup
app_state: dict = {}

# In-memory storage — defined early so lifespan can reference it
stocks_db = {
    "AAPL": {"symbol": "AAPL", "price": 150.25, "change": 2.50, "change_percent": 1.69},
    "GOOGL": {"symbol": "GOOGL", "price": 140.80, "change": -1.20, "change_percent": -0.84},
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the LlamaIndex agent once at startup and share it across requests."""
    try:
        app_state["agent"] = build_agent(stocks_db)
        print("✅ LlamaIndex agent ready")
    except ValueError as e:
        print(f"⚠️  Agent not started: {e}")
        app_state["agent"] = None
    yield
    app_state.clear()

app = FastAPI(
    title="Stocksight API",
    description="A simple stock market API powered by FastAPI + LlamaIndex",
    version="2.0.0",
    lifespan=lifespan,
)

class Stock(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float

class StockUpdate(BaseModel):
    price: float
    change: float
    change_percent: float

@app.get("/")
def read_root():
    return {"message": "Welcome to Stocksight API", "version": "1.0.0"}

@app.get("/stocks")
def get_all_stocks():
    """Get all stocks"""
    return list(stocks_db.values())

@app.get("/stocks/{symbol}")
def get_stock(symbol: str):
    """Get a specific stock by symbol"""
    symbol = symbol.upper()
    if symbol not in stocks_db:
        return {"error": f"Stock {symbol} not found"}, 404
    return stocks_db[symbol]

@app.post("/stocks")
def create_stock(stock: Stock):
    """Create a new stock"""
    if stock.symbol in stocks_db:
        return {"error": f"Stock {stock.symbol} already exists"}, 400
    stocks_db[stock.symbol] = stock.dict()
    return {"message": "Stock created successfully", "stock": stock}

@app.put("/stocks/{symbol}")
def update_stock(symbol: str, stock_update: StockUpdate):
    """Update an existing stock"""
    symbol = symbol.upper()
    if symbol not in stocks_db:
        return {"error": f"Stock {symbol} not found"}, 404
    stocks_db[symbol].update(stock_update.dict())
    return {"message": "Stock updated successfully", "stock": stocks_db[symbol]}

@app.delete("/stocks/{symbol}")
def delete_stock(symbol: str):
    """Delete a stock"""
    symbol = symbol.upper()
    if symbol not in stocks_db:
        return {"error": f"Stock {symbol} not found"}, 404
    deleted = stocks_db.pop(symbol)
    return {"message": "Stock deleted successfully", "stock": deleted}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


class AgentQuery(BaseModel):
    question: str


@app.post("/agent/query")
def agent_query(body: AgentQuery):
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
            detail="Agent is not available. Check that OPENAI_API_KEY is set in your .env file.",
        )
    response = agent.chat(body.question)
    return {"question": body.question, "answer": str(response)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
