from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class OpenAIGenerationConfig(GenerationConfig):
    """OpenAI config. `model` is an OpenAI model id such as "gpt-4.1" or "o4-mini".

    Reasoning depth, parallel tool calls and strict JSON schemas use the generic
    fields of `GenerationConfig` and the adapter's own rules, so this class adds
    no field of its own.
    """


"""
sources: https://developers.openai.com/api/docs/models
         https://developers.openai.com/api/docs/guides/reasoning
         https://developers.openai.com/api/docs/guides/function-calling
         https://developers.openai.com/api/docs/guides/structured-outputs
         https://developers.openai.com/api/reference/resources/chat (exact request schema; ChatOpenAI uses /v1/chat/completions, not /v1/responses)
"""
