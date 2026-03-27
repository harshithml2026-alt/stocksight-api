import random

_RESPONSES = [
    "Nice try, but my MBA didn't cover that. Ask me about stocks instead.",
    "That's a great question... for literally any other assistant. I only do finance.",
    "I've read thousands of SEC filings and not one mentioned that. Try again.",
    "Error 404: Finance relevance not found. Want to try a stock question?",
    "I'm flattered you think I'm a general genius, but I'm more of a Wall Street nerd.",
    "My circuits are wired for balance sheets, not that. Ask me something financial!",
    "Bold strategy — asking a financial assistant about that. Doesn't quite pay off though.",
    "I only speak fluent Finance. That question was in a different language entirely.",
    "That's above my pay grade. My pay grade is strictly stocks, markets, and SEC filings.",
    "I've seen riskier trades, but asking me that is up there. Stick to finance questions!",
    "Warren Buffett once said 'invest in what you know.' I know finance. Ask me about that.",
    "My portfolio doesn't include that topic. Diversification has its limits.",
    "I ran the numbers and the ROI on answering that is zero. Finance questions only!",
    "Interesting pivot — but I'm not that kind of assistant. Let's talk markets instead.",
    "That question just got rejected like a bad IPO. Finance only, please.",
    "I tried to find a financial angle on that. Couldn't. Ask me about earnings instead.",
    "Due diligence complete: that's outside my scope. Stocks and filings are my thing.",
    "If that question were a stock, I'd short it. I only go long on finance topics.",
    "You're testing me like a stress test on a bank. I hold firm — finance questions only!",
    "Plot twist: I only know about money. Ask me something about markets or SEC filings!",
]


def get_off_topic_response() -> str:
    return random.choice(_RESPONSES)
