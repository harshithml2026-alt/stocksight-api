import os
from dotenv import load_dotenv
from pymongo.collection import Collection
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

load_dotenv()

# ── Stock tool functions (sync, pymongo) ──────────────────────────────────────
# These run synchronously inside LlamaIndex's agent loop.
# They use the same MongoDB database as the async FastAPI routes,
# but through a separate synchronous pymongo client.

def get_stock(symbol: str, col: Collection) -> dict:
    """Get current price and change data for a specific stock symbol."""
    doc = col.find_one({"symbol": symbol.upper()}, {"_id": 0})
    return doc if doc else {"error": f"Stock '{symbol.upper()}' not found"}


def list_stocks(col: Collection) -> list:
    """List all available stocks with their current prices."""
    return list(col.find({}, {"_id": 0}))


def get_top_gainer(col: Collection) -> dict:
    """Find the stock with the highest percentage gain today."""
    doc = col.find_one(sort=[("change_percent", -1)], projection={"_id": 0})
    return doc if doc else {"error": "No stocks available"}


def get_top_loser(col: Collection) -> dict:
    """Find the stock with the biggest loss today."""
    doc = col.find_one(sort=[("change_percent", 1)], projection={"_id": 0})
    return doc if doc else {"error": "No stocks available"}


def market_summary(col: Collection) -> dict:
    """Return a high-level market summary: total, gainers, losers, flat."""
    total = col.count_documents({})
    if total == 0:
        return {"error": "No stocks available"}
    gainers = col.count_documents({"change_percent": {"$gt": 0}})
    losers  = col.count_documents({"change_percent": {"$lt": 0}})
    return {
        "total":      total,
        "gainers":    gainers,
        "losers":     losers,
        "flat":       total - gainers - losers,
        "top_gainer": col.find_one(sort=[("change_percent", -1)], projection={"_id": 0}),
        "top_loser":  col.find_one(sort=[("change_percent",  1)], projection={"_id": 0}),
    }


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent(col: Collection) -> OpenAIAgent:
    """
    Build a LlamaIndex OpenAI agent wired up with MongoDB stock tools.

    Args:
        col: A synchronous pymongo Collection for the 'stocks' collection.

    Returns:
        A ready-to-use OpenAIAgent instance.
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
            fn=lambda symbol: get_stock(symbol, col),
            name="get_stock",
            description="Get the current price and change data for a specific stock symbol.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: list_stocks(col),
            name="list_stocks",
            description="List all available stocks with their current prices.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: get_top_gainer(col),
            name="get_top_gainer",
            description="Find the stock with the highest percentage gain today.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: get_top_loser(col),
            name="get_top_loser",
            description="Find the stock with the biggest loss today.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: market_summary(col),
            name="market_summary",
            description="Get a high-level market summary: total stocks, gainers, losers, flat.",
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
