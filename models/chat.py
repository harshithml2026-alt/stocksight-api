from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: datetime


class SessionSummary(BaseModel):
    """Lightweight session info returned in list views."""
    id: str
    preview: str
    createdAt: int = Field(..., description="Unix timestamp in milliseconds")


class SessionDetail(BaseModel):
    """Full session with all messages."""
    id: str
    ip_address: str
    preview: str
    createdAt: int
    updatedAt: int
    messages: list[ChatMessage]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict] = []
