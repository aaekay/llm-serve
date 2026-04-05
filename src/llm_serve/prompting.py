from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

from llm_serve.schemas import ChatMessage, ContentPart


_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL)


def extract_text_content(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            continue
        if hasattr(part, "text"):
            parts.append(part.text)
    return " ".join(parts)


def normalize_messages_for_chat_template(messages: List[Union[ChatMessage, Dict[str, object]]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip()
            content = extract_text_content(message.get("content", ""))
        else:
            role = message.role
            content = extract_text_content(message.content)
        normalized.append({"role": role, "content": content})
    return normalized


def render_messages_to_prompt(messages: List[ChatMessage]) -> str:
    rendered = []
    for message in normalize_messages_for_chat_template(messages):
        rendered.append("%s: %s" % (message["role"].upper(), message["content"]))
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def split_reasoning_output(text: str) -> tuple[Optional[str], str]:
    reasoning_parts = [match.group(1).strip() for match in _THINK_BLOCK_RE.finditer(text) if match.group(1).strip()]
    if not reasoning_parts:
        return None, text.strip()
    answer = _THINK_BLOCK_RE.sub("", text).strip()
    reasoning = "\n\n".join(reasoning_parts).strip() or None
    return reasoning, answer


class ThinkingContentStripper:
    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    def push(self, chunk: str) -> List[str]:
        if not chunk:
            return []
        self._buffer += chunk
        emitted: List[str] = []

        while self._buffer:
            if self._inside_think:
                close_index = self._buffer.find(_THINK_CLOSE_TAG)
                if close_index == -1:
                    self._buffer = self._buffer[-(len(_THINK_CLOSE_TAG) - 1):]
                    break
                self._buffer = self._buffer[close_index + len(_THINK_CLOSE_TAG):].lstrip("\r\n")
                self._inside_think = False
                continue

            open_index = self._buffer.find(_THINK_OPEN_TAG)
            if open_index == -1:
                keep = len(_THINK_OPEN_TAG) - 1
                if len(self._buffer) <= keep:
                    break
                emit_upto = len(self._buffer) - keep
                emitted.append(self._buffer[:emit_upto])
                self._buffer = self._buffer[emit_upto:]
                break

            if open_index > 0:
                emitted.append(self._buffer[:open_index])
            self._buffer = self._buffer[open_index + len(_THINK_OPEN_TAG):]
            self._inside_think = True

        return [part for part in emitted if part]

    def finish(self) -> str:
        if self._inside_think:
            return ""
        tail = self._buffer
        self._buffer = ""
        return tail
