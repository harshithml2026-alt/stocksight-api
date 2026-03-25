import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase

_PRIVATE_RANGES = ("127.", "10.", "192.168.", "172.", "::1", "localhost")


async def lookup_ip_locations(ips: list[str]) -> dict[str, str]:
    """Batch-lookup IP locations via ip-api.com (free, no API key required)."""
    public_ips = [ip for ip in ips if not any(ip.startswith(p) for p in _PRIVATE_RANGES)]
    result = {ip: "Private / Local" for ip in ips if ip not in public_ips}

    if not public_ips:
        return result

    payload = [{"query": ip, "fields": "status,city,country,query"} for ip in public_ips]
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post("http://ip-api.com/batch", json=payload)
            for item in resp.json():
                ip = item.get("query", "")
                if item.get("status") == "success":
                    city = item.get("city", "")
                    country = item.get("country", "")
                    result[ip] = f"{city}, {country}" if city else country
                else:
                    result[ip] = "Unknown"
    except Exception:
        for ip in public_ips:
            result.setdefault(ip, "Unavailable")

    return result


class AdminService:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self.col = db["chat_sessions"]

    async def get_ips_paginated(
        self, page: int, page_size: int,
        sort_by: str = "last_active", sort_dir: str = "desc"
    ) -> dict:
        """Group all sessions by IP address with counts and activity timestamps."""
        skip = (page - 1) * page_size
        sort_field = sort_by if sort_by in ("session_count", "first_seen", "last_active") else "last_active"
        sort_order = -1 if sort_dir == "desc" else 1

        pipeline = [
            {
                "$group": {
                    "_id": "$ip_address",
                    "session_count": {"$sum": 1},
                    "last_active": {"$max": "$updated_at_ms"},
                    "first_seen": {"$min": "$created_at_ms"},
                }
            },
            {"$sort": {sort_field: sort_order}},
            {"$facet": {
                "items": [{"$skip": skip}, {"$limit": page_size}],
                "total": [{"$count": "count"}],
            }},
        ]

        result = await self.col.aggregate(pipeline).to_list(1)
        data = result[0] if result else {"items": [], "total": []}
        total = data["total"][0]["count"] if data["total"] else 0

        return {
            "items": [
                {
                    "ip_address": doc["_id"],
                    "session_count": doc["session_count"],
                    "last_active": doc["last_active"],
                    "first_seen": doc["first_seen"],
                }
                for doc in data["items"]
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    async def get_sessions_for_ip(self, ip_address: str, page: int, page_size: int) -> dict:
        """Return paginated sessions for a specific IP address."""
        skip = (page - 1) * page_size
        total = await self.col.count_documents({"ip_address": ip_address})

        cursor = self.col.find(
            {"ip_address": ip_address},
            {"_id": 0, "session_id": 1, "preview": 1, "created_at_ms": 1,
             "updated_at_ms": 1, "is_archived": 1},
        ).sort("updated_at_ms", -1).skip(skip).limit(page_size)

        docs = await cursor.to_list(None)
        return {
            "items": [
                {
                    "id": doc["session_id"],
                    "preview": doc["preview"],
                    "createdAt": doc["created_at_ms"],
                    "updatedAt": doc["updated_at_ms"],
                    "isArchived": doc.get("is_archived", False),
                }
                for doc in docs
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    async def get_stats(self) -> dict:
        """Overall counts for the dashboard header."""
        total_sessions = await self.col.count_documents({})
        total_ips_result = await self.col.aggregate([
            {"$group": {"_id": "$ip_address"}},
            {"$count": "count"},
        ]).to_list(1)
        archived = await self.col.count_documents({"is_archived": True})
        return {
            "total_sessions": total_sessions,
            "total_ips": total_ips_result[0]["count"] if total_ips_result else 0,
            "archived_sessions": archived,
            "active_sessions": total_sessions - archived,
        }
