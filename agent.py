import os
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

load_dotenv()

# ---------------------------------------------------------------------------
# Stock tools — these wrap the same in-memory DB used by the FastAPI routes
# ---------------------------------------------------------------------------

def get_stock(symbol: str, stocks_db: dict) -> dict:
    """Get current price and change data for a specific stock symbol."""
    symbol = symbol.upper()
    if symbol not in stocks_db:
        return {"error": f"Stock '{symbol}' not found"}
    return stocks_db[symbol]


def list_stocks(stocks_db: dict) -> list:
    """List all available stocks with their current prices."""
    return list(stocks_db.values())


def get_top_gainer(stocks_db: dict) -> dict:
    """Find the stock with the highest percentage change today."""
    if not stocks_db:
        return {"error": "No stocks available"}
    return max(stocks_db.values(), key=lambda s: s["change_percent"])


def get_top_loser(stocks_db: dict) -> dict:
    """Find the stock with the lowest (most negative) percentage change today."""
    if not stocks_db:
        return {"error": "No stocks available"}
    return min(stocks_db.values(), key=lambda s: s["change_percent"])


def market_summary(stocks_db: dict) -> dict:
    """Return a high-level summary of the current market: gainers, losers, flat."""
    if not stocks_db:
        return {"error": "No stocks available"}
    gainers = [s for s in stocks_db.values() if s["change_percent"] > 0]
    losers  = [s for s in stocks_db.values() if s["change_percent"] < 0]
    flat    = [s for s in stocks_db.values() if s["change_percent"] == 0]
    return {
        "total": len(stocks_db),
        "gainers": len(gainers),
        "losers": len(losers),
        "flat": len(flat),
        "top_gainer": max(gainers, key=lambda s: s["change_percent"]) if gainers else None,
        "top_loser":  min(losers,  key=lambda s: s["change_percent"]) if losers  else None,
    }


# ---------------------------------------------------------------------------
# Agent factory — call this once and reuse the returned agent
# ---------------------------------------------------------------------------

def build_agent(stocks_db: dict) -> OpenAIAgent:
    """
    Build and return a LlamaIndex OpenAI agent wired up with stock tools.

    Args:
        stocks_db: The shared in-memory dict from main.py

    Returns:
        A ready-to-use OpenAIAgent instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    llm = OpenAI(model=model, api_key=api_key)

    tools = [
        FunctionTool.from_defaults(
            fn=lambda symbol: get_stock(symbol, stocks_db),
            name="get_stock",
            description="Get the current price and change data for a specific stock symbol.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: list_stocks(stocks_db),
            name="list_stocks",
            description="List all available stocks with their current prices.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: get_top_gainer(stocks_db),
            name="get_top_gainer",
            description="Find the stock with the highest percentage gain today.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: get_top_loser(stocks_db),
            name="get_top_loser",
            description="Find the stock with the biggest loss today.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: market_summary(stocks_db),
            name="market_summary",
            description="Get a high-level market summary: number of gainers, losers, and flat stocks.",
        ),
    ]

    agent = OpenAIAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are a helpful stock market assistant. "
            "Use the available tools to answer questions about stocks accurately. "
            "Always cite the data returned by the tools."
        ),
    )

    return agent
