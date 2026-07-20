from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class MistralGenerationConfig(GenerationConfig):
    model: str = "mistral-medium-latest"
    """Mistral model identifier, e.g. "mistral-medium-latest" or "mistral-large-latest"."""

    json_schema_name: str = "response"
    """Name reported for the JSON_SCHEMA response format."""

    json_schema_strict: bool = True
    """Enforce strict JSON Schema conformance for the JSON_SCHEMA response format."""


"""
sources: https://docs.mistral.ai/getting-started/models/models_overview/
         https://docs.mistral.ai/capabilities/structured_output/custom
"""
