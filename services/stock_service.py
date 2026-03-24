from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument
from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional

from models.stock import StockCreate, StockUpdate


def _serialize(doc: dict) -> dict:
    """Convert a MongoDB document to a JSON-serialisable dict (str id)."""
    return {"id": str(doc["_id"]), **{k: v for k, v in doc.items() if k != "_id"}}


class StockService:
    """Async CRUD service for the `stocks` collection."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self.col = db["stocks"]

    # ── CREATE ─────────────────────────────────────────────────────────────────

    async def create(self, payload: StockCreate) -> dict:
        """Insert a new stock document and return the created record."""
        now = datetime.now(timezone.utc)
        doc = payload.model_dump()
        doc["symbol"] = doc["symbol"].upper()
        doc["created_at"] = now
        doc["updated_at"] = now
        result = await self.col.insert_one(doc)
        created = await self.col.find_one({"_id": result.inserted_id})
        return _serialize(created)

    # ── READ ───────────────────────────────────────────────────────────────────

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[dict]:
        """Return a paginated list of all stocks."""
        return [
            _serialize(doc)
            async for doc in self.col.find().skip(skip).limit(limit)
        ]

    async def get_by_symbol(self, symbol: str) -> Optional[dict]:
        """Find a stock by its ticker symbol (case-insensitive)."""
        doc = await self.col.find_one({"symbol": symbol.upper()})
        return _serialize(doc) if doc else None

    async def get_by_id(self, stock_id: str) -> Optional[dict]:
        """Find a stock by its MongoDB ObjectId string."""
        if not ObjectId.is_valid(stock_id):
            return None
        doc = await self.col.find_one({"_id": ObjectId(stock_id)})
        return _serialize(doc) if doc else None

    async def search(self, query: str) -> list[dict]:
        """Case-insensitive regex search across symbol, name, and sector."""
        regex = {"$regex": query, "$options": "i"}
        cursor = self.col.find(
            {"$or": [{"symbol": regex}, {"name": regex}, {"sector": regex}]}
        )
        return [_serialize(doc) async for doc in cursor]

    # ── UPDATE ─────────────────────────────────────────────────────────────────

    async def update(self, symbol: str, payload: StockUpdate) -> Optional[dict]:
        """Partial update — only the supplied fields are changed."""
        data = {k: v for k, v in payload.model_dump().items() if v is not None}
        if not data:
            return await self.get_by_symbol(symbol)
        data["updated_at"] = datetime.now(timezone.utc)
        doc = await self.col.find_one_and_update(
            {"symbol": symbol.upper()},
            {"$set": data},
            return_document=ReturnDocument.AFTER,
        )
        return _serialize(doc) if doc else None

    # ── DELETE ─────────────────────────────────────────────────────────────────

    async def delete(self, symbol: str) -> Optional[dict]:
        """Delete a stock by symbol and return the deleted record."""
        doc = await self.col.find_one_and_delete({"symbol": symbol.upper()})
        return _serialize(doc) if doc else None

    # ── ANALYTICS ─────────────────────────────────────────────────────────────

    async def get_top_gainer(self) -> Optional[dict]:
        """Stock with the highest change_percent today."""
        doc = await self.col.find_one(sort=[("change_percent", -1)])
        return _serialize(doc) if doc else None

    async def get_top_loser(self) -> Optional[dict]:
        """Stock with the most negative change_percent today."""
        doc = await self.col.find_one(sort=[("change_percent", 1)])
        return _serialize(doc) if doc else None

    async def market_summary(self) -> dict:
        """Aggregate overview: total stocks, gainers, losers, flat."""
        total   = await self.col.count_documents({})
        gainers = await self.col.count_documents({"change_percent": {"$gt": 0}})
        losers  = await self.col.count_documents({"change_percent": {"$lt": 0}})
        return {
            "total":      total,
            "gainers":    gainers,
            "losers":     losers,
            "flat":       total - gainers - losers,
            "top_gainer": await self.get_top_gainer(),
            "top_loser":  await self.get_top_loser(),
        }
