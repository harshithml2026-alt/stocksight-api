import uuid
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _serialize_session(doc: dict) -> dict:
    """Map MongoDB doc → API shape expected by the UI."""
    return {
        "id": doc["session_id"],
        "preview": doc["preview"],
        "createdAt": doc["created_at_ms"],
    }


def _serialize_session_detail(doc: dict) -> dict:
    return {
        "id": doc["session_id"],
        "ip_address": doc["ip_address"],
        "preview": doc["preview"],
        "createdAt": doc["created_at_ms"],
        "updatedAt": doc["updated_at_ms"],
        "messages": doc.get("messages", []),
    }


class ChatService:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self.col = db["chat_sessions"]

    async def create_session(self, ip_address: str, first_question: str) -> str:
        """Create a new session and return its session_id."""
        session_id = str(uuid.uuid4())
        now_ms = _now_ms()
        await self.col.insert_one({
            "session_id": session_id,
            "ip_address": ip_address,
            "preview": first_question[:80],
            "created_at_ms": now_ms,
            "updated_at_ms": now_ms,
            "messages": [],
        })
        return session_id

    async def append_messages(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        metrics: dict | None = None,
        sources: list | None = None,
    ) -> None:
        """Append a user + assistant message pair to an existing session."""
        now = datetime.now(timezone.utc).isoformat()
        now_ms = _now_ms()
        assistant_msg = {"role": "assistant", "content": assistant_text, "timestamp": now}
        if metrics:
            assistant_msg["metrics"] = metrics
        if sources:
            # Deduplicate by filing identity, keep top 5 unique filings
            seen, deduped = set(), []
            for s in sources:
                meta = s.get("metadata", {})
                key = (
                    meta.get("ticker") or meta.get("company", ""),
                    meta.get("filing_type") or meta.get("form_type", ""),
                    meta.get("period_of_report") or meta.get("date", ""),
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(s)
                if len(deduped) == 5:
                    break
            assistant_msg["sources"] = deduped
        await self.col.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_text, "timestamp": now},
                            assistant_msg,
                        ]
                    }
                },
                "$set": {"updated_at_ms": now_ms},
            },
        )

    async def get_sessions_by_ip(self, ip_address: str) -> list[dict]:
        """Return all non-archived sessions for an IP sorted newest-first."""
        cursor = self.col.find(
            {"ip_address": ip_address, "is_archived": {"$ne": True}},
            {"_id": 0, "session_id": 1, "preview": 1, "created_at_ms": 1},
        ).sort("created_at_ms", -1)
        return [_serialize_session(doc) async for doc in cursor]

    async def archive_session(self, session_id: str) -> bool:
        """Soft-delete a session by marking it as archived."""
        result = await self.col.update_one(
            {"session_id": session_id},
            {"$set": {"is_archived": True, "updated_at_ms": _now_ms()}},
        )
        return result.modified_count > 0

    async def get_session(self, session_id: str) -> dict | None:
        """Return a full session document with all messages."""
        doc = await self.col.find_one({"session_id": session_id}, {"_id": 0})
        return _serialize_session_detail(doc) if doc else None

    async def session_belongs_to_ip(self, session_id: str, ip_address: str) -> bool:
        doc = await self.col.find_one(
            {"session_id": session_id, "ip_address": ip_address}, {"_id": 1}
        )
        return doc is not None

    async def delete_session(self, session_id: str) -> bool:
        result = await self.col.delete_one({"session_id": session_id})
        return result.deleted_count > 0
