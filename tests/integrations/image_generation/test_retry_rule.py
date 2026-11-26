"""The retry rule of DALL-E, seen through its public methods.

A generation that fails after the request left may have been done and billed,
so only a 429 or a 503 (the service rejected the request before working on it)
costs more than one call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr

from agent_platform.core.errors import ProviderError
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.image_generation.dalle.config import DalleConfig

if TYPE_CHECKING:
    from agent_platform.integrations.image_generation.dalle.provider import (
        DallEImageGenerator,
    )

_REQUEST = httpx.Request("POST", "https://vendor.test/v1")
_OPERATION = "Image generation failed"
_DALLE_PROVIDER = "agent_platform.integrations.image_generation.dalle.provider"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


def _openai_status(status: int) -> Exception:
    openai = pytest.importorskip("openai")
    error_class = {
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        503: openai.InternalServerError,
    }[status]
    return error_class("vendor said no", response=_response(status), body=None)


def _dalle(mocker, error: Exception) -> tuple[DallEImageGenerator, AsyncMock]:
    pytest.importorskip("openai")
    from agent_platform.integrations.image_generation.dalle.provider import (
        DallEImageGenerator,
    )

    client_class = mocker.patch(f"{_DALLE_PROVIDER}.AsyncOpenAI")
    sdk_call = AsyncMock(side_effect=error)
    client_class.return_value.images.generate = sdk_call
    generator = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("sk-test")))
    return generator, sdk_call


class TestDallERetryRule:
    @pytest.mark.parametrize("status", [429, 503])
    async def test_a_rejected_request_costs_three_calls(
        self, status, mocker, no_retry_sleep
    ):
        generator, sdk_call = _dalle(mocker, _openai_status(status))

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert sdk_call.call_count == 3
        assert caught.value.retryable is True
        assert str(caught.value).startswith(_OPERATION)

    async def test_a_server_error_costs_one_call(self, mocker, no_retry_sleep):
        generator, sdk_call = _dalle(mocker, _openai_status(500))

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert sdk_call.call_count == 1
        assert caught.value.retryable is False
        assert str(caught.value).startswith(_OPERATION)

    async def test_a_timeout_costs_one_call(self, mocker, no_retry_sleep):
        timeout = pytest.importorskip("openai").APITimeoutError(request=_REQUEST)
        generator, sdk_call = _dalle(mocker, timeout)

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert sdk_call.call_count == 1
        assert caught.value.retryable is False
        assert str(caught.value).startswith(_OPERATION)

    async def test_generate_many_follows_the_same_rule(self, mocker, no_retry_sleep):
        generator, sdk_call = _dalle(mocker, _openai_status(503))

        with pytest.raises(ProviderError):
            await generator.generate_many("a cat", n=1)

        assert sdk_call.call_count == 3

    async def test_an_invalid_size_is_never_sent_nor_retried(
        self, mocker, no_retry_sleep, caplog
    ):
        generator, sdk_call = _dalle(mocker, _openai_status(503))

        with caplog.at_level(logging.WARNING, logger="agent_platform"):
            with pytest.raises(ProviderError) as caught:
                await generator.generate("a cat", size="512x512")

        assert sdk_call.call_count == 0
        assert caught.value.retryable is False
        assert not [r for r in caplog.records if "attempt" in r.getMessage()]

    def test_the_sdk_does_not_retry_and_has_a_300_second_default_timeout(self):
        pytest.importorskip("openai")
        from agent_platform.integrations.image_generation.dalle.provider import (
            DEFAULT_IMAGE_TIMEOUT_SECONDS,
            DallEImageGenerator,
        )

        generator = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("sk-test")))

        kwargs = generator._client_kwargs(DalleConfig())

        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == DEFAULT_IMAGE_TIMEOUT_SECONDS == 300.0

    def test_a_configured_timeout_wins_over_the_default(self):
        pytest.importorskip("openai")
        from agent_platform.integrations.image_generation.dalle.provider import (
            DallEImageGenerator,
        )

        generator = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("sk-test")))

        assert generator._client_kwargs(DalleConfig(timeout=12.0))["timeout"] == 12.0
