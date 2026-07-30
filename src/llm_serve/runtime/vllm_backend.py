"""Backward-compatible name for the unified supervised vLLM backend."""

from .autotune import ManualVLLMBackend


VLLMModelBackend = ManualVLLMBackend
