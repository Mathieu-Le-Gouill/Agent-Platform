from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.fallback import (
    FallbackEntry,
    FallbackLLMProvider,
)
from agent_platform.core.resilience import CircuitState, RateLimiter
from agent_platform.core.schemas.message import Prompt
from agent_platform.integrations.llm.openai.provider import OpenAILLM
from tests.helpers import (
    make_fake_llm_response,
    make_fake_stream,
    make_text_stream_chunks,
)


def _mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.agenerate = AsyncMock()
    provider.stream = MagicMock()
    return provider


@pytest.fixture
def prompt() -> Prompt:
    return Prompt.build(system="hi")


class TestConstruction:
    def test_requires_at_least_one_entry(self):
        with pytest.raises(ValueError, match="at least one entry"):
            FallbackLLMProvider([])


class TestAgenerate:
    @pytest.mark.asyncio
    async def test_uses_primary_when_it_succeeds(self, prompt):
        primary = _mock_provider()
        primary.agenerate.return_value = make_fake_llm_response("primary")
        secondary = _mock_provider()

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        result = await chain.agenerate(prompt)

        assert result.message.text == "primary"
        secondary.agenerate.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_fails(self, prompt):
        primary = _mock_provider()
        primary.agenerate.side_effect = RuntimeError("boom")
        secondary = _mock_provider()
        secondary.agenerate.return_value = make_fake_llm_response("secondary")

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        result = await chain.agenerate(prompt)

        assert result.message.text == "secondary"

    @pytest.mark.asyncio
    async def test_passes_each_entrys_own_model(self, prompt):
        primary = _mock_provider()
        primary.agenerate.side_effect = RuntimeError("boom")
        secondary = _mock_provider()
        secondary.agenerate.return_value = make_fake_llm_response("secondary")

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        await chain.agenerate(prompt)

        args, _ = secondary.agenerate.call_args
        assert args[1].model == "model-b"

    @pytest.mark.asyncio
    async def test_raises_provider_error_when_all_fail(self, prompt):
        primary = _mock_provider()
        primary.agenerate.side_effect = RuntimeError("boom-a")
        secondary = _mock_provider()
        secondary.agenerate.side_effect = RuntimeError("boom-b")

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        with pytest.raises(ProviderError, match="fallback chain"):
            await chain.agenerate(prompt)

    @pytest.mark.asyncio
    async def test_skips_entry_with_open_circuit(self, prompt):
        primary = _mock_provider()
        primary.agenerate.side_effect = RuntimeError("boom")
        secondary = _mock_provider()
        secondary.agenerate.return_value = make_fake_llm_response("secondary")

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )
        chain._circuits[0]._state = CircuitState.OPEN
        chain._circuits[0]._opened_at = float("inf")

        result = await chain.agenerate(prompt)

        assert result.message.text == "secondary"
        primary.agenerate.assert_not_called()


class TestCircuitCounting:
    """Only a retryable failure counts against an entry's circuit; any failure falls through."""

    @staticmethod
    def _chain(primary: MagicMock, secondary: MagicMock) -> FallbackLLMProvider:
        return FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

    @staticmethod
    def _failing_stream(error: Exception):
        def _raise(*args, **kwargs):
            raise error

        return _raise

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("retryable", "expected_state"),
        [(False, CircuitState.CLOSED), (True, CircuitState.OPEN)],
    )
    async def test_agenerate(self, prompt, retryable, expected_state):
        primary = _mock_provider()
        primary.agenerate.side_effect = ProviderError("bad", retryable=retryable)
        secondary = _mock_provider()
        secondary.agenerate.return_value = make_fake_llm_response("secondary")
        chain = self._chain(primary, secondary)

        for _ in range(5):
            result = await chain.agenerate(prompt)
            assert result.message.text == "secondary"

        assert chain._circuits[0].state is expected_state

    @pytest.mark.parametrize(
        ("retryable", "expected_state"),
        [(False, CircuitState.CLOSED), (True, CircuitState.OPEN)],
    )
    def test_generate(self, prompt, retryable, expected_state):
        primary = _mock_provider()
        primary.generate.side_effect = ProviderError("bad", retryable=retryable)
        secondary = _mock_provider()
        secondary.generate.return_value = make_fake_llm_response("secondary")
        chain = self._chain(primary, secondary)

        for _ in range(5):
            assert chain.generate(prompt).message.text == "secondary"

        assert chain._circuits[0].state is expected_state

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("retryable", "expected_state"),
        [(False, CircuitState.CLOSED), (True, CircuitState.OPEN)],
    )
    async def test_stream(self, prompt, retryable, expected_state):
        primary = _mock_provider()
        primary.stream.side_effect = self._failing_stream(
            ProviderError("bad", retryable=retryable)
        )
        secondary = _mock_provider()
        secondary.stream.side_effect = lambda *a, **kw: make_fake_stream(
            make_text_stream_chunks(["ok"])
        )
        chain = self._chain(primary, secondary)

        for _ in range(5):
            chunks = [c async for c in chain.stream(prompt)]
            assert [c.delta for c in chunks if c.delta] == ["ok"]

        assert chain._circuits[0].state is expected_state


class TestStreamedVendorErrors:
    @pytest.mark.asyncio
    async def test_streamed_bad_requests_leave_every_circuit_closed(self, prompt):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        bad_request = openai.BadRequestError(
            "bad", response=httpx.Response(400, request=request), body=None
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=bad_request)
        primary = OpenAILLM(model="gpt-4.1")
        primary._stream_client = lambda config: client
        secondary = _mock_provider()
        secondary.stream.side_effect = lambda *a, **kw: make_fake_stream(
            make_text_stream_chunks(["ok"])
        )
        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="gpt-4.1"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        for _ in range(5):
            chunks = [c async for c in chain.stream(prompt)]
            assert [c.delta for c in chunks if c.delta] == ["ok"]

        assert [c.state for c in chain._circuits] == [CircuitState.CLOSED] * 2


class TestAllFailed:
    @staticmethod
    def _chain(*providers: MagicMock) -> FallbackLLMProvider:
        return FallbackLLMProvider(
            [
                FallbackEntry(provider=p, model=f"model-{i}")
                for i, p in enumerate(providers)
            ]
        )

    @staticmethod
    def _rate_limited() -> ProviderError:
        return ProviderError(
            "limited", code="llm_rate_limited", retryable=True, retry_after=7
        )

    def _assert_keeps_the_rate_limit(self, error: ProviderError) -> None:
        assert "All providers in the fallback chain failed" in str(error)
        assert error.code == "llm_rate_limited"
        assert error.retryable is True
        assert error.retry_after == 7

    @pytest.mark.asyncio
    async def test_agenerate_keeps_the_last_errors_contract(self, prompt):
        first, last = _mock_provider(), _mock_provider()
        first.agenerate.side_effect = ProviderError("a", code="llm_timeout")
        last.agenerate.side_effect = self._rate_limited()

        with pytest.raises(ProviderError) as caught:
            await self._chain(first, last).agenerate(prompt)

        self._assert_keeps_the_rate_limit(caught.value)

    def test_generate_keeps_the_last_errors_contract(self, prompt):
        only = _mock_provider()
        only.generate.side_effect = self._rate_limited()

        with pytest.raises(ProviderError) as caught:
            self._chain(only).generate(prompt)

        self._assert_keeps_the_rate_limit(caught.value)

    @pytest.mark.asyncio
    async def test_stream_keeps_the_last_errors_contract(self, prompt):
        only = _mock_provider()

        def _raise(*args, **kwargs):
            raise self._rate_limited()

        only.stream.side_effect = _raise

        with pytest.raises(ProviderError) as caught:
            _ = [c async for c in self._chain(only).stream(prompt)]

        self._assert_keeps_the_rate_limit(caught.value)

    @pytest.mark.asyncio
    async def test_an_unclassified_failure_is_unavailable(self, prompt):
        only = _mock_provider()
        only.agenerate.side_effect = RuntimeError("boom")

        with pytest.raises(ProviderError) as caught:
            await self._chain(only).agenerate(prompt)

        assert caught.value.code == "provider_unavailable"

    @pytest.mark.asyncio
    async def test_open_circuits_raise_a_retryable_unavailable_error(self, prompt):
        only = _mock_provider()
        chain = self._chain(only)
        chain._circuits[0]._state = CircuitState.OPEN
        chain._circuits[0]._opened_at = float("inf")

        with pytest.raises(ProviderError) as caught:
            await chain.agenerate(prompt)

        assert caught.value.code == "provider_unavailable"
        assert caught.value.retryable is True
        only.agenerate.assert_not_called()


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_agenerate_acquires_from_rate_limiter(self, prompt):
        primary = _mock_provider()
        primary.agenerate.return_value = make_fake_llm_response("primary")
        rate_limiter = RateLimiter(rate=100.0)
        rate_limiter.acquire = AsyncMock(wraps=rate_limiter.acquire)

        chain = FallbackLLMProvider(
            [FallbackEntry(provider=primary, model="model-a")],
            rate_limiter=rate_limiter,
        )

        await chain.agenerate(prompt)

        rate_limiter.acquire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_acquires_from_rate_limiter(self, prompt):
        primary = _mock_provider()
        primary.stream.side_effect = lambda *a, **kw: make_fake_stream(
            make_text_stream_chunks(["ok"])
        )
        rate_limiter = RateLimiter(rate=100.0)
        rate_limiter.acquire = AsyncMock(wraps=rate_limiter.acquire)

        chain = FallbackLLMProvider(
            [FallbackEntry(provider=primary, model="model-a")],
            rate_limiter=rate_limiter,
        )

        _ = [c async for c in chain.stream(prompt)]

        rate_limiter.acquire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_rate_limiter_by_default(self, prompt):
        primary = _mock_provider()
        primary.agenerate.return_value = make_fake_llm_response("primary")

        chain = FallbackLLMProvider([FallbackEntry(provider=primary, model="model-a")])

        result = await chain.agenerate(prompt)

        assert result.message.text == "primary"


class TestStream:
    @pytest.mark.asyncio
    async def test_falls_back_before_first_chunk(self, prompt):
        primary = _mock_provider()

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        primary.stream.side_effect = _raise
        secondary = _mock_provider()
        secondary.stream.side_effect = lambda *a, **kw: make_fake_stream(
            make_text_stream_chunks(["ok"])
        )

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        chunks = [c async for c in chain.stream(prompt)]

        assert [c.delta for c in chunks if c.delta] == ["ok"]

    @pytest.mark.asyncio
    async def test_does_not_switch_providers_mid_stream(self, prompt):
        async def _failing_stream():
            yield make_text_stream_chunks(["partial"])[0]
            raise RuntimeError("dropped connection")

        primary = _mock_provider()
        primary.stream.side_effect = lambda *a, **kw: _failing_stream()
        secondary = _mock_provider()

        chain = FallbackLLMProvider(
            [
                FallbackEntry(provider=primary, model="model-a"),
                FallbackEntry(provider=secondary, model="model-b"),
            ]
        )

        with pytest.raises(RuntimeError, match="dropped connection"):
            _ = [c async for c in chain.stream(prompt)]

        secondary.stream.assert_not_called()
