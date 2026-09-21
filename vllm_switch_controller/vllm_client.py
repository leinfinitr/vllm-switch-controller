"""Compatibility imports for the former vLLM-specific client module."""

from .backends import EngineControlError as VLLMClientError
from .engine_client import EngineClient as VLLMClient
from .engine_client import filter_end_to_end_headers

__all__ = ["VLLMClient", "VLLMClientError", "filter_end_to_end_headers"]
