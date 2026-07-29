from __future__ import annotations

from agent_platform.core.config import ModelConfig
from agent_platform.core.interfaces.llm.response import ResponseFormat


class GenerationConfig(ModelConfig):
    # Sampling randomness: 0 is near-deterministic, higher values increase diversity.
    # None leaves it unset so each provider's own native default applies.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
    temperature: float | None = None
    # Upper bound on tokens generated in the completion.
    max_tokens: int | None = None
    # Nucleus sampling: restricts sampling to the smallest token set whose cumulative probability exceeds top_p.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p
    top_p: float | None = None
    # Restricts sampling to the top_k most likely tokens at each step; provider-specific support.
    top_k: int | None = None
    # Sequences that, once generated, stop further token generation.
    stop_sequences: list[str] = []
    # Fixes the sampling RNG for (best-effort) reproducible outputs; provider-specific support.
    seed: int | None = None
    # Penalizes tokens proportionally to how often they've already appeared, discouraging repetition.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-frequency_penalty
    frequency_penalty: float | None = None
    # Penalizes tokens that have appeared at all so far, encouraging new topics.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-presence_penalty
    presence_penalty: float | None = None
    # Maximum time in seconds to wait for a response before aborting the request.
    timeout: float | None = None
    # Maximum number of retry attempts on transient/provider errors.
    max_retries: int | None = None
    # Desired shape of the model output (e.g. plain text vs. JSON); provider-specific meaning, see each provider's config.
    response_format: ResponseFormat = ResponseFormat.TEXT
    # JSON Schema the output must conform to when response_format requests structured JSON.
    json_schema: dict | None = None
