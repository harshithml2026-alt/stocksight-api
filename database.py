import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL   = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "stocksightDB")


# ── Async client — used by FastAPI routes ──────────────────────────────────────

class _AsyncMongoDB:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None

_async_db = _AsyncMongoDB()


async def connect_db() -> None:
    _async_db.client = AsyncIOMotorClient(MONGODB_URL)
    _async_db.db = _async_db.client[DATABASE_NAME]
    # Ensure unique index on ticker symbol
    await _async_db.db.stocks.create_index("symbol", unique=True)
    print(f"✅ Connected to MongoDB [{DATABASE_NAME}] (async)")


async def close_db() -> None:
    if _async_db.client:
        _async_db.client.close()
        print("🔌 MongoDB async connection closed")


def get_db() -> AsyncIOMotorDatabase:
    """Return the async database instance (for use inside async FastAPI routes)."""
    return _async_db.db


# ── Sync client — used by LlamaIndex agent tools ───────────────────────────────

class _SyncMongoDB:
    client: MongoClient = None
    db = None

_sync_db = _SyncMongoDB()


def connect_sync_db() -> None:
    _sync_db.client = MongoClient(MONGODB_URL)
    _sync_db.db = _sync_db.client[DATABASE_NAME]
    print(f"✅ Connected to MongoDB [{DATABASE_NAME}] (sync)")


def close_sync_db() -> None:
    if _sync_db.client:
        _sync_db.client.close()
        print("🔌 MongoDB sync connection closed")


def get_sync_collection(name: str) -> Collection:
    """Return a synchronous pymongo Collection (for use inside agent tools)."""
    return _sync_db.db[name]
