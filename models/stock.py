from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class StockBase(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────────
    symbol: str = Field(
        ..., min_length=1, max_length=10, description="Ticker symbol e.g. AAPL"
    )
    name: str = Field(..., description="Company full name e.g. Apple Inc.")
    exchange: Optional[str] = Field(
        None, description="Exchange where listed e.g. NASDAQ, NYSE, LSE"
    )
    sector: Optional[str] = Field(
        None, description="Business sector e.g. Technology, Healthcare, Finance"
    )
    industry: Optional[str] = Field(
        None, description="Industry sub-category e.g. Consumer Electronics, Semiconductors"
    )
    currency: str = Field("USD", description="Trading currency code e.g. USD, EUR, GBP")

    # ── Price data ────────────────────────────────────────────────────────────
    price: float = Field(..., ge=0, description="Current / last traded price")
    open_price: Optional[float] = Field(None, ge=0, description="Day opening price")
    close_price: Optional[float] = Field(
        None, ge=0, description="Previous session closing price"
    )
    high: Optional[float] = Field(None, ge=0, description="Day high")
    low: Optional[float] = Field(None, ge=0, description="Day low")
    week_52_high: Optional[float] = Field(None, ge=0, description="52-week high")
    week_52_low: Optional[float] = Field(None, ge=0, description="52-week low")

    # ── Volume & market cap ───────────────────────────────────────────────────
    volume: Optional[int] = Field(None, ge=0, description="Current session trading volume")
    avg_volume: Optional[int] = Field(None, ge=0, description="Average daily trading volume")
    market_cap: Optional[float] = Field(
        None, ge=0, description="Market capitalisation in USD"
    )

    # ── Fundamental metrics ───────────────────────────────────────────────────
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    eps: Optional[float] = Field(None, description="Earnings per share (TTM)")
    dividend_yield: Optional[float] = Field(
        None, ge=0, description="Annual dividend yield as a percentage"
    )
    beta: Optional[float] = Field(
        None, description="Beta – stock volatility relative to the market"
    )

    # ── Change ────────────────────────────────────────────────────────────────
    change: float = Field(0.0, description="Absolute price change from previous close")
    change_percent: float = Field(0.0, description="Percentage price change")

    # ── Media ─────────────────────────────────────────────────────────────────
    image_url: Optional[str] = Field(None, description="URL to the company logo or stock image")

    # ── Status ────────────────────────────────────────────────────────────────
    is_active: bool = Field(True, description="Whether this stock is actively tracked")


class StockCreate(StockBase):
    """Payload for POST /stocks – creates a new stock document."""
    pass


class StockUpdate(BaseModel):
    """
    Payload for PUT /stocks/{symbol}.
    All fields are optional; only the fields you supply are updated.
    """
    name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    open_price: Optional[float] = Field(None, ge=0)
    close_price: Optional[float] = Field(None, ge=0)
    high: Optional[float] = Field(None, ge=0)
    low: Optional[float] = Field(None, ge=0)
    week_52_high: Optional[float] = Field(None, ge=0)
    week_52_low: Optional[float] = Field(None, ge=0)
    volume: Optional[int] = Field(None, ge=0)
    avg_volume: Optional[int] = Field(None, ge=0)
    market_cap: Optional[float] = Field(None, ge=0)
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = Field(None, ge=0)
    beta: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    image_url: Optional[str] = None
    is_active: Optional[bool] = None


class StockResponse(StockBase):
    """Full stock document returned from the database, including metadata."""
    id: str = Field(..., description="MongoDB document ID")
    created_at: datetime = Field(..., description="When the record was first created")
    updated_at: datetime = Field(..., description="When the record was last updated")

    model_config = ConfigDict(from_attributes=True)
