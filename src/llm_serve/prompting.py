from __future__ import annotations

from typing import List

from llm_serve.schemas import ChatMessage


def render_messages_to_prompt(messages: List[ChatMessage]) -> str:
    rendered = []
    for message in messages:
        rendered.append("%s: %s" % (message.role.upper(), message.content))
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)
