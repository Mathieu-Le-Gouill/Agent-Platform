from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.fallback import (
    FallbackEntry,
    FallbackLLMProvider,
)
from agent_platform.core.resilience import CircuitState
from agent_platform.core.schemas.message import Prompt
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
