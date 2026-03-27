import os
import time
from openai import OpenAI
from pinecone import Pinecone
from services.guardrails import check_input, check_output
from data.instructions import SYSTEM_PROMPT_RAG, SYSTEM_PROMPT_DIRECT, _today
from data.off_topic_responses import get_off_topic_response

_openai: OpenAI = None
_index = None

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 20

# Tickers that have embeddings in Pinecone
SUPPORTED_TICKERS = {"AAPL", "GOOGL", "AMZN", "META", "TSLA", "MSFT", "NVDA"}


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
    from datetime import date as _date
    current_year = _date.today().year
    last_year = current_year - 1

    messages = [
        {
            "role": "system",
            "content": (
                f"Today is {_date.today().strftime('%B %d, %Y')}. "
                f"'Last year' = {last_year}. 'This year' = {current_year}. "
                "Resolve all relative time references to absolute years.\n\n"
                "You are a router. Be lenient with minor spelling mistakes or typos in company names "
                "and tickers (e.g. 'Nvdia' → NVDA, 'Amazn' → AMZN, 'Microsft' → MSFT, 'APPL' → AAPL).\n"
                "If the latest message is a short reaction or acknowledgement (e.g. 'wow', 'ok', 'nice', "
                "'interesting', 'really?') and the prior conversation was about finance, reply GENERAL.\n"
                "Given a question, reply with ONLY one of these tokens — nothing else:\n"
                "  OFFTOPIC  — completely unrelated to finance, stocks, markets, or economics (e.g. cooking, sports, weather, coding). NEVER use for investment questions — those are GENERAL.\n"
                "  GENERAL   — any finance, economics, or investing question with no specific supported company (e.g. 'best stock to buy', 'should I invest in stocks', 'what is a P/E ratio', 'how do markets work')\n"
                "  UNKNOWN   — about a company NOT in this list: AAPL, GOOGL, AMZN, META, TSLA, MSFT, NVDA\n"
                "  <TICKER>        — e.g. NVDA, MSFT, META (no year)\n"
                "  <TICKER>:<YEAR> — e.g. NVDA:2024, META:2025 (year mentioned or resolved)\n\n"
                "Company name → ticker mapping:\n"
                "  Apple, Apple Inc. → AAPL\n"
                "  Google, Alphabet, Alphabet Inc. → GOOGL\n"
                "  Amazon, Amazon.com → AMZN\n"
                "  Meta, Meta Platforms, Facebook → META\n"
                "  Tesla, Tesla Inc. → TSLA\n"
                "  Microsoft → MSFT\n"
                "  NVIDIA, Nvidia → NVDA\n\n"
                "Examples:\n"
                "  'What is a P/E ratio?' → GENERAL\n"
                "  'How do I cook pasta?' → OFFTOPIC\n"
                "  'What is the best stock to buy?' → GENERAL\n"
                "  'Should I invest in stocks?' → GENERAL\n"
                "  'What is Apple revenue?' → AAPL\n"
                "  'How much did Meta spend on buybacks?' → META\n"
                "  'What did Google earn in 2023?' → GOOGL:2023\n"
                f"  'What did NVIDIA spend on R&D last year?' → NVDA:{last_year}\n"
                "  'How is Tesla doing this year?' → TSLA:{current_year}\n"
                "  'What did Netflix earn?' → UNKNOWN"
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

    # Normalize year — strip non-numeric prefix (e.g. "FY2024" → "2024")
    year = None
    if len(parts) == 2:
        import re as _re
        year_match = _re.search(r"\d{4}", parts[1])
        year = year_match.group() if year_match else None

    if route not in SUPPORTED_TICKERS and route not in ("GENERAL", "OFFTOPIC", "UNKNOWN"):
        route = "OFFTOPIC"

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
    is_safe, block_reason = check_input(question, history)
    if not is_safe:
        return {"question": question, "answer": block_reason, "sources": []}

    # 2. Classify question, extract ticker and year in one LLM call
    route, year = _classify_question(question, history)

    if route == "OFFTOPIC":
        return {"question": question, "answer": get_off_topic_response(), "sources": []}

    if route == "GENERAL":
        return _query_llm_direct(question, history)

    if route == "UNKNOWN":
        supported_list = "AAPL (Apple), GOOGL (Alphabet), AMZN (Amazon), META (Meta), TSLA (Tesla), MSFT (Microsoft), NVDA (NVIDIA)"
        return {
            "question": question,
            "answer": (
                f"I currently only have SEC filing data for: {supported_list}. "
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

    # If no year was specified, keep only chunks from the most recent year present
    # to avoid mixing data across multiple years in the same answer
    if not year:
        years_present = sorted(
            {m.get("metadata", {}).get("year") for m in matches if m.get("metadata", {}).get("year")},
            reverse=True,
        )
        if years_present:
            most_recent = years_present[0]
            matches = [m for m in matches if m.get("metadata", {}).get("year") == most_recent]

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
