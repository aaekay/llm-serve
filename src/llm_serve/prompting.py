from __future__ import annotations

from typing import List, Union

from llm_serve.schemas import ChatMessage, ContentPart


def _extract_text(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if hasattr(part, "text"):
            parts.append(part.text)
    return " ".join(parts)


def render_messages_to_prompt(messages: List[ChatMessage]) -> str:
    rendered = []
    for message in messages:
        rendered.append("%s: %s" % (message.role.upper(), _extract_text(message.content)))
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)
