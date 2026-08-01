from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.resilience import CircuitBreaker, CircuitState, RateLimiter
from agent_platform.core.schemas.message import Prompt

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

__all__ = ["FallbackEntry", "FallbackLLMProvider"]


@dataclass(frozen=True)
class FallbackEntry:
    """One provider in a fallback chain, paired with the model it should be
    called with (vendors don't share model ids, so this can't be a single
    `model` string applied uniformly across the chain)."""

    provider: BaseLLMProvider
    model: str


class FallbackLLMProvider(BaseLLMProvider):
    """Tries `entries` in order, skipping any whose circuit is open, falling
    through to the next on failure.

    A `BaseLLMProvider` itself, so it's a drop-in replacement anywhere a
    single provider is injected today (`Agent(llm=...)`,
    `config/container.py::build_agent`) - callers don't need to know a call
    is backed by more than one vendor. Each entry gets its own
    `CircuitBreaker` so one vendor's outage doesn't count against another's,
    and a persistently-failing entry stops being tried (instead of paying its
    latency on every call) until its reset timeout elapses.
    """

    def __init__(
        self,
        entries: Sequence[FallbackEntry],
        *,
        circuit_breaker_factory: type[CircuitBreaker] = CircuitBreaker,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        if not entries:
            raise ValueError("FallbackLLMProvider requires at least one entry")
        self._entries = list(entries)
        self._circuits = [circuit_breaker_factory() for _ in entries]
        self._rate_limiter = rate_limiter

    def _config_for(
        self, config: GenerationConfig | None, model: str
    ) -> GenerationConfig:
        return (config or GenerationConfig()).model_copy(update={"model": model})

    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        last_exc: Exception | None = None
        for entry, circuit in zip(self._entries, self._circuits, strict=True):
            if circuit.state is CircuitState.OPEN:
                continue
            try:
                return circuit.call(
                    entry.provider.generate,
                    prompt,
                    self._config_for(config, entry.model),
                    tools,
                )
            except Exception as exc:  # noqa: BLE001 - any provider failure falls through
                last_exc = exc
                continue
        raise ProviderError(
            "All providers in the fallback chain failed", retryable=False
        ) from last_exc

    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        last_exc: Exception | None = None
        for entry, circuit in zip(self._entries, self._circuits, strict=True):
            if circuit.state is CircuitState.OPEN:
                continue
            try:
                return await circuit.acall(
                    entry.provider.agenerate,
                    prompt,
                    self._config_for(config, entry.model),
                    tools,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        raise ProviderError(
            "All providers in the fallback chain failed", retryable=False
        ) from last_exc

    async def stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Falls over to the next entry only if a provider fails *before*
        yielding its first chunk. Once a chunk has reached the caller,
        switching providers mid-stream would mix output from two different
        completions, so a failure past that point is raised as-is instead of
        silently retried."""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        last_exc: Exception | None = None
        for entry, circuit in zip(self._entries, self._circuits, strict=True):
            if circuit.state is CircuitState.OPEN:
                continue
            yielded_any = False
            try:
                async for chunk in entry.provider.stream(
                    prompt, self._config_for(config, entry.model), tools
                ):
                    yielded_any = True
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                circuit.record_failure()
                if yielded_any:
                    raise
                last_exc = exc
                continue
            circuit.record_success()
            return
        raise ProviderError(
            "All providers in the fallback chain failed", retryable=False
        ) from last_exc
