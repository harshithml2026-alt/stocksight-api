import os
import time
from openai import OpenAI
from pinecone import Pinecone
from services.guardrails import check_input, check_output
from data.instructions import SYSTEM_PROMPT_RAG, SYSTEM_PROMPT_DIRECT, _today

_openai: OpenAI = None
_index = None

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 20

# Tickers that have embeddings in Pinecone
SUPPORTED_TICKERS = {"NVDA", "MSFT"}


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        _openai = OpenAI(api_key=api_key)
    return _openai


def _get_index():
    global _index
    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set.")
        index_name = os.getenv("PINECONE_INDEX_NAME", "stocksight-data-ingestion")
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(index_name)
    return _index


def _embed(text: str) -> list[float]:
    response = _get_openai().embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def _extract_text(metadata: dict) -> str:
    """Extract the chunk text from Pinecone metadata regardless of field name."""
    for key in ("text", "chunk_text", "content", "page_content"):
        if key in metadata:
            return metadata[key]
    # Fallback: join all string values
    return " ".join(str(v) for v in metadata.values() if isinstance(v, str))


def fetch_context(question: str, top_k: int = TOP_K) -> dict:
    """
    Embed the question and return raw Pinecone chunks without calling OpenAI.
    Use this to inspect what context is being retrieved.
    """
    query_vector = _embed(question)
    index = _get_index()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    matches = results.get("matches", [])
    chunks = []
    for match in matches:
        meta = match.get("metadata", {})
        chunks.append({
            "id": match.get("id"),
            "score": round(match.get("score", 0), 4),
            "text": _extract_text(meta),
            "metadata": meta,
        })

    return {
        "question": question,
        "top_k": top_k,
        "total_retrieved": len(chunks),
        "chunks": chunks,
    }


def _classify_question(
    question: str, history: list[dict] | None = None
) -> tuple[str, str | None]:
    """
    Single LLM call that classifies the question, extracts the ticker, and
    extracts an explicit year if mentioned. Includes the last few conversation
    turns so follow-up questions resolve correctly.

    Returns a tuple (route, year) where:
      route — one of: "NONE", "NVDA", "MSFT", "UNKNOWN"
      year  — 4-digit string (e.g. "2025") or None if not mentioned

    Output format from the LLM: "TICKER:YEAR" or just "TICKER".
    Examples: "NVDA:2025", "MSFT:2023", "NVDA", "NONE", "UNKNOWN"
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    supported = ", ".join(sorted(SUPPORTED_TICKERS))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a router for a financial assistant. "
                "Identify whether the user's latest question (considering the conversation history) "
                "is about a specific stock or company, and if so, which ticker it maps to "
                f"from this supported list: {supported}. "
                "Also extract the year if the user mentions a specific year.\n\n"
                "Reply with exactly one token in this format (no other text):\n"
                "- NONE              — question is not about any specific company or stock\n"
                "- UNKNOWN           — question is about a company/stock NOT in the supported list\n"
                "- NVDA              — about NVIDIA, no specific year mentioned\n"
                "- MSFT              — about Microsoft, no specific year mentioned\n"
                "- NVDA:2025         — about NVIDIA, year 2025 mentioned\n"
                "- MSFT:2024         — about Microsoft, year 2024 mentioned\n"
                "(Replace year with the actual year from the question)"
            ),
        }
    ]

    # Include last 4 turns (2 exchanges) so follow-up questions resolve correctly
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": question})

    completion = _get_openai().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=10,
        temperature=0,
    )
    raw = completion.choices[0].message.content.strip().upper()

    # Parse "TICKER:YEAR" or "TICKER"
    parts = raw.split(":")
    route = parts[0].strip()
    year = parts[1].strip() if len(parts) == 2 and parts[1].strip().isdigit() else None

    if route not in SUPPORTED_TICKERS and route not in ("NONE", "UNKNOWN"):
        route = "NONE"

    return route, year


def _query_llm_direct(
    question: str,
    history: list[dict] | None = None,
) -> dict:
    """Call the LLM directly without Pinecone context (for non-stock questions)."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages = [{"role": "system", "content": SYSTEM_PROMPT_DIRECT.format(today=_today())}]
    for turn in (history or []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    t0 = time.perf_counter()
    completion = _get_openai().chat.completions.create(model=model, messages=messages)
    elapsed = time.perf_counter() - t0

    usage = completion.usage
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0
    tokens_per_sec = round(completion_tokens / elapsed, 2) if elapsed > 0 else 0

    _, safe_answer = check_output(completion.choices[0].message.content)
    return {
        "question": question,
        "answer": safe_answer,
        "sources": [],
        "metrics": {
            "total_tokens": total_tokens,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": completion_tokens,
            "inference_time_sec": round(elapsed, 3),
            "tokens_per_sec": tokens_per_sec,
        },
    }


def query(
    question: str,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """
    Route the question:
      - Not stock-related  → call LLM directly (no Pinecone)
      - Known ticker       → query Pinecone filtered to that ticker, then call LLM
      - Unknown ticker     → tell user we only have NVDA and MSFT data

    Args:
        question: The current user question.
        top_k: Number of Pinecone chunks to retrieve (only used for supported tickers).
        history: Prior messages as [{"role": "user"|"assistant", "content": "..."}].
    """
    # 1. Guardrail — check input for injection and harmful content
    is_safe, block_reason = check_input(question)
    if not is_safe:
        return {"question": question, "answer": block_reason, "sources": []}

    # 2. Classify question, extract ticker and year in one LLM call
    route, year = _classify_question(question, history)

    if route == "NONE":
        return _query_llm_direct(question, history)

    if route == "UNKNOWN":
        supported = " and ".join(sorted(SUPPORTED_TICKERS))
        return {
            "question": question,
            "answer": (
                f"I currently only have SEC filing data for {supported}. "
                "Support for more tickers is coming soon! "
                "In the meantime, feel free to ask about those companies or ask a general financial question."
            ),
            "sources": [],
        }

    # route is a supported ticker (e.g. "NVDA" or "MSFT")
    ticker = route

    # 3. Embed query
    query_vector = _embed(question)

    # 4. Fetch relevant chunks from Pinecone — filtered to ticker and year (if specified)
    pinecone_filter: dict = {"ticker": {"$eq": ticker}}
    if year:
        pinecone_filter["year"] = {"$eq": year}

    index = _get_index()
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter,
    )

    matches = results.get("matches", [])
    if not matches:
        scope = f"{ticker} ({year})" if year else ticker
        return {
            "question": question,
            "answer": f"No relevant context found in the SEC filings index for {scope}.",
            "sources": [],
        }

    # 4. Build context string and source list
    context_parts = []
    sources = []
    for match in matches:
        meta = match.get("metadata", {})
        chunk = _extract_text(meta)
        if chunk:
            context_parts.append(chunk)
        sources.append({
            "id": match.get("id"),
            "score": round(match.get("score", 0), 4),
            "metadata": {k: v for k, v in meta.items()
                         if k not in ("text", "chunk_text", "content", "page_content")},
        })

    context = "\n\n---\n\n".join(context_parts)

    # 5. Build message list: system → history → current question with context
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages = [{"role": "system", "content": SYSTEM_PROMPT_RAG.format(today=_today())}]

    # Append prior turns so the model has full conversation context
    for turn in (history or []):
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current question with freshly retrieved context
    messages.append({
        "role": "user",
        "content": f"Context from SEC filings:\n\n{context}\n\nQuestion: {question}",
    })

    t0 = time.perf_counter()
    completion = _get_openai().chat.completions.create(model=model, messages=messages)
    elapsed = time.perf_counter() - t0

    usage = completion.usage
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0
    tokens_per_sec = round(completion_tokens / elapsed, 2) if elapsed > 0 else 0

    _, safe_answer = check_output(completion.choices[0].message.content)
    return {
        "question": question,
        "answer": safe_answer,
        "sources": sources,
        "metrics": {
            "total_tokens": total_tokens,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": completion_tokens,
            "inference_time_sec": round(elapsed, 3),
            "tokens_per_sec": tokens_per_sec,
        },
    }
