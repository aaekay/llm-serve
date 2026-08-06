from __future__ import annotations

import json
import os
import sys

import httpx


BASE_URL = os.environ.get("LLM_SERVE_BASE_URL", "http://127.0.0.1:11424/v1").rstrip("/")
MODEL = os.environ.get("LLM_SERVE_MODEL", "Qwen/Qwen3.5-27B")
TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_directory",
        "description": "Return the current working directory.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
}


def show(label: str, payload: object) -> None:
    print("\n=== %s ===" % label, flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def main() -> int:
    print("Testing llm-serve at %s with %s" % (BASE_URL, MODEL), flush=True)
    with httpx.Client(base_url=BASE_URL, timeout=300) as client:
        models = client.get("/models")
        models.raise_for_status()
        show("Models", models.json())

        chat = client.post(
            "/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "developer", "content": "Use the supplied tool."},
                    {"role": "user", "content": "What is the current working directory?"},
                ],
                "tools": [TOOL],
                "tool_choice": "required",
                "parallel_tool_calls": True,
                "max_tokens": 512,
            },
        )
        chat.raise_for_status()
        chat_payload = chat.json()
        show("Chat tool call", chat_payload)
        tool_calls = chat_payload.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            raise RuntimeError("Chat Completions did not return a tool call")

        response = client.post(
            "/responses",
            json={
                "model": MODEL,
                "input": "Call get_current_directory to answer the question.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_current_directory",
                        "description": "Return the current working directory.",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    }
                ],
                "tool_choice": "required",
                "max_output_tokens": 512,
            },
        )
        response.raise_for_status()
        response_payload = response.json()
        show("Responses tool call", response_payload)
        if not any(item.get("type") == "function_call" for item in response_payload.get("output", [])):
            raise RuntimeError("Responses API did not return a function_call item")

    print("\nAgent provider smoke test passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nAgent provider smoke test failed: %s" % exc, file=sys.stderr, flush=True)
        raise
