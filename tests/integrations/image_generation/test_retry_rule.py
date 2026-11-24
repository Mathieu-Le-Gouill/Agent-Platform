"""The retry rule of DALL-E and Midjourney, seen through their public methods.

A generation that fails after the request left may have been done and billed,
so only a 429 or a 503 (the service rejected the request before working on it)
costs more than one call. A failed Midjourney download never sends the
`/imagine` request again.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from pydantic import SecretStr

from agent_platform.core.errors import ProviderError
from agent_platform.integrations.credentials import (
    MidjourneyCredentials,
    OpenAICredentials,
)
from agent_platform.integrations.image_generation.dalle.config import DalleConfig
from agent_platform.integrations.image_generation.dalle.provider import (
    DEFAULT_IMAGE_TIMEOUT_SECONDS,
    DallEImageGenerator,
)
from agent_platform.integrations.image_generation.midjourney.config import (
    MidjourneyConfig,
)
from agent_platform.integrations.image_generation.midjourney.provider import (
    MidjourneyGenerator,
)

_REQUEST = httpx.Request("POST", "https://vendor.test/v1")
_OPERATION = "Image generation failed"
_DALLE_PROVIDER = "agent_platform.integrations.image_generation.dalle.provider"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


def _openai_status(status: int) -> openai.APIStatusError:
    error_class = {
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        503: openai.InternalServerError,
    }[status]
    return error_class("vendor said no", response=_response(status), body=None)


def _dalle(mocker, error: Exception) -> tuple[DallEImageGenerator, AsyncMock]:
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
        timeout = openai.APITimeoutError(request=_REQUEST)
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
        generator = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("sk-test")))

        kwargs = generator._client_kwargs(DalleConfig())

        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == DEFAULT_IMAGE_TIMEOUT_SECONDS == 300.0

    def test_a_configured_timeout_wins_over_the_default(self):
        generator = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("sk-test")))

        assert generator._client_kwargs(DalleConfig(timeout=12.0))["timeout"] == 12.0


def _http_status(status: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        "vendor said no", request=_REQUEST, response=_response(status)
    )


def _midjourney(
    mocker, *, post_error: Exception | None = None, get_error: Exception | None = None
) -> tuple[MidjourneyGenerator, AsyncMock, AsyncMock, MagicMock]:
    client_class = mocker.patch("httpx.AsyncClient")
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client_class.return_value = client

    accepted = MagicMock()
    accepted.json.return_value = {"image_url": "https://cdn.example.com/i.png"}
    client.post = AsyncMock(side_effect=post_error, return_value=accepted)
    downloaded = MagicMock()
    downloaded.content = b"bytes"
    client.get = AsyncMock(side_effect=get_error, return_value=downloaded)

    generator = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("k")))
    return generator, client.post, client.get, client_class


class TestMidjourneyRetryRule:
    async def test_a_503_on_imagine_costs_three_posts(self, mocker, no_retry_sleep):
        generator, post, _, _ = _midjourney(mocker, post_error=_http_status(503))

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert post.call_count == 3
        assert caught.value.retryable is True
        assert str(caught.value).startswith(_OPERATION)

    async def test_a_500_on_imagine_costs_one_post(self, mocker, no_retry_sleep):
        generator, post, _, _ = _midjourney(mocker, post_error=_http_status(500))

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert post.call_count == 1
        assert caught.value.retryable is False
        assert str(caught.value).startswith(_OPERATION)

    async def test_a_timeout_on_imagine_costs_one_post(self, mocker, no_retry_sleep):
        timeout = httpx.ReadTimeout("slow", request=_REQUEST)
        generator, post, _, _ = _midjourney(mocker, post_error=timeout)

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert post.call_count == 1
        assert caught.value.retryable is False

    async def test_generate_many_follows_the_same_rule(self, mocker, no_retry_sleep):
        generator, post, _, _ = _midjourney(mocker, post_error=_http_status(503))

        with pytest.raises(ProviderError):
            await generator.generate_many("a cat", n=2)

        assert post.call_count == 3

    async def test_a_failed_download_never_posts_again(self, mocker, no_retry_sleep):
        generator, post, get, _ = _midjourney(mocker, get_error=_http_status(503))

        with pytest.raises(ProviderError) as caught:
            await generator.generate("a cat")

        assert post.call_count == 1
        assert get.call_count == 3
        assert str(caught.value).startswith(_OPERATION)

    async def test_the_default_timeout_stays_120_seconds(self, mocker, no_retry_sleep):
        generator, _, _, client_class = _midjourney(mocker)

        await generator.generate("a cat", config=MidjourneyConfig())

        assert client_class.call_args.kwargs["timeout"] == 120
