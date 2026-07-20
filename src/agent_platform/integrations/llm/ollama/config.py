from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class OllamaGenerationConfig(GenerationConfig):
    model: str = "llama3.2"
    """Ollama model tag to run, e.g. "llama3.2" or "mistral"."""

    repeat_penalty: float | None = None
    """Ollama-native repetition control: multiplicative, default 1.1, symmetric around
    1.0 — not the same semantics as OpenAI/Anthropic-style additive `frequency_penalty`."""

    mirostat: int | None = None
    """Mirostat sampling mode: 0 disabled, 1 Mirostat, 2 Mirostat 2.0."""

    mirostat_tau: float | None = None
    """Mirostat target entropy (coherence vs diversity balance)."""

    mirostat_eta: float | None = None
    """Mirostat learning rate."""

    num_ctx: int | None = None
    """Context window size in tokens."""


# sources: https://github.com/ollama/ollama/blob/main/docs/api.md
