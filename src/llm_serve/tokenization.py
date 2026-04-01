from __future__ import annotations

import math
from typing import Iterable


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    # Approximate 1 token ~= 4 characters for a broad set of LLM tokenizers.
    return max(1, int(math.ceil(len(text) / 4.0)))


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    total = 0
    for message in messages:
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_text_tokens(message.get("content", ""))
        total += 4
    return total
