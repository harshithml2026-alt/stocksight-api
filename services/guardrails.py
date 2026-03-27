import re
import os
from openai import OpenAI

_openai: OpenAI = None


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        _openai = OpenAI(api_key=api_key)
    return _openai


# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|directives?|context)",
    r"forget\s+(previous|prior|your)\s+(instructions?|prompts?|context|rules?)",
    r"disregard\s+(previous|prior|your|all)\s*(instructions?|prompts?|rules?)?",
    r"override\s+(your\s+)?(instructions?|system|prompt)",
    r"you\s+are\s+now\s+\w",
    r"act\s+as\s+(if\s+you('re|\s+are)\s+|a\s+)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"new\s+(instructions?|system\s+prompt|directive|persona)",
    r"(system|user|assistant)\s*:\s*(you|ignore|forget|disregard)",
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"jailbreak",
    r"\bdan\b.*\bmode\b",
    r"developer\s+mode",
    r"do\s+anything\s+now",
    r"prompt\s+injection",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

# ── Output leakage / injection patterns ──────────────────────────────────────
_OUTPUT_LEAKAGE_PATTERNS = [
    r"my\s+(system\s+)?instructions?\s+(say|are|tell|state)",
    r"i\s+was\s+(told|instructed|programmed)\s+to",
    r"as\s+per\s+my\s+(system\s+)?instructions?",
    r"ignore\s+(previous|prior|above)\s+(instructions?|prompts?)",
    r"my\s+programming\s+(says?|tells?|instructs?)",
    r"you\s+can\s+now\s+ask\s+me\s+to",
    r"i\s+am\s+now\s+operating\s+(in|as)",
]

_OUTPUT_LEAKAGE_RE = re.compile(
    "|".join(_OUTPUT_LEAKAGE_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)


def check_input(text: str) -> tuple[bool, str]:
    """
    Validate user input before processing.

    Runs three checks in order:
      1. Prompt injection — regex scan for override/jailbreak patterns
      2. Harmful content  — OpenAI Moderation API
      3. Finance relevance — LLM binary check (max_tokens=1)

    Returns:
        (True, "")           — input is safe, proceed
        (False, reason_str)  — input blocked, reason_str is the user-facing message
    """
    # 1. Prompt injection check (fast, no API call)
    if _INJECTION_RE.search(text):
        return False, "Your message contains content that cannot be processed. Please rephrase your question."

    # 2. Harmful content check via OpenAI Moderation API (free endpoint)
    result = _get_openai().moderations.create(input=text)
    outcome = result.results[0]
    if outcome.flagged:
        flagged_categories = [
            cat for cat, flagged in outcome.categories.model_dump().items() if flagged
        ]
        return False, (
            "Your message was flagged for inappropriate content "
            f"({', '.join(flagged_categories).replace('/', ' / ')}). "
            "Please keep questions relevant to financial topics."
        )

    # 3. Finance relevance check — block non-finance questions before any routing
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    completion = _get_openai().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a topic classifier for a financial assistant. "
                    "Your job is to classify the TOPIC of the question only — not whether you can answer it. "
                    "Reply YES if the question topic is related to finance, stocks, markets, "
                    "economics, investing, SEC filings, earnings, or any of these companies:\n"
                    "  Apple (AAPL), Alphabet / Google (GOOGL), Amazon (AMZN), "
                    "Meta / Facebook (META), Tesla (TSLA), Microsoft (MSFT), NVIDIA (NVDA).\n"
                    "Reply NO for everything else (cooking, sports, weather, coding, "
                    "general knowledge, personal advice, etc.).\n"
                    "Examples:\n"
                    "  'How is Nvidia doing in 2026?' → YES\n"
                    "  'What did Meta spend on buybacks?' → YES\n"
                    "  'How does Amazon price its cloud services?' → YES\n"
                    "  'How do I cook pasta?' → NO\n"
                    "  'What is the weather today?' → NO"
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=3,
        temperature=0,
    )
    answer = completion.choices[0].message.content.strip().upper()
    if not answer.startswith("YES"):
        from data.off_topic_responses import get_off_topic_response
        return False, get_off_topic_response()

    return True, ""


def check_output(text: str) -> tuple[bool, str]:
    """
    Validate LLM output before returning it to the user.

    Scans for system prompt leakage or injected instructions in the response.

    Returns:
        (True, text)                — output is safe, return as-is
        (False, safe_fallback_str)  — output is unsafe, return the fallback instead
    """
    if _OUTPUT_LEAKAGE_RE.search(text):
        return False, "I'm sorry, I wasn't able to generate a safe response. Please try rephrasing your question."

    return True, text
