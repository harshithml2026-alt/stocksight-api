import os
import time
from openai import OpenAI
from pinecone import Pinecone

_openai: OpenAI = None
_index = None

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 20


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


def query(
    question: str,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """
    Embed the question, retrieve top-k chunks from Pinecone,
    then generate an answer with OpenAI using the retrieved context
    and the full session conversation history.

    Args:
        question: The current user question.
        top_k: Number of Pinecone chunks to retrieve.
        history: Prior messages as [{"role": "user"|"assistant", "content": "..."}].
    """
    # 1. Embed query
    query_vector = _embed(question)

    # 2. Fetch relevant chunks from Pinecone
    index = _get_index()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    matches = results.get("matches", [])
    if not matches:
        return {
            "question": question,
            "answer": "No relevant context found in the SEC filings index.",
            "sources": [],
        }

    # 3. Build context string and source list
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

    # 4. Build message list: system → history → current question with context
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assistant. "
                "Answer the user's question using the SEC filing context provided "
                "and the conversation history. "
                "Be precise and cite specific details from the context. "
                "If the context does not contain enough information, say so clearly."
            ),
        }
    ]

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

    return {
        "question": question,
        "answer": completion.choices[0].message.content,
        "sources": sources,
        "metrics": {
            "total_tokens": total_tokens,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": completion_tokens,
            "inference_time_sec": round(elapsed, 3),
            "tokens_per_sec": tokens_per_sec,
        },
    }
