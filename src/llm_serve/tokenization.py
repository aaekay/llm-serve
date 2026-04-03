from __future__ import annotations

import math
from typing import Iterable, List, Union


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    # Approximate 1 token ~= 4 characters for a broad set of LLM tokenizers.
    return max(1, int(math.ceil(len(text) / 4.0)))


def _extract_content_text(content: Union[str, List[dict], None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return " ".join(parts)


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    total = 0
    for message in messages:
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_text_tokens(_extract_content_text(message.get("content", "")))
        total += 4
    return total
