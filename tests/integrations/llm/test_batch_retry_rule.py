"""The retry rule of OpenAI and Anthropic batches, seen through their public methods.

A batch submission is never retried: a failure after the request left may mean
the batch was created and will be billed, and a second submission would run it
twice. Reading a batch is idempotent, so a 503 is retried.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.batch import BatchRequest
from agent_platform.core.schemas.message import Prompt, UserMessage
from agent_platform.integrations.credentials import (
    AnthropicCredentials,
    OpenAICredentials,
)

_REQUEST = httpx.Request("POST", "https://vendor.test/v1")
_REQUESTS = [
    BatchRequest(custom_id="r1", prompt=Prompt(messages=[UserMessage(content="Hi")]))
]


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


def _openai_503() -> Exception:
    import openai

    return openai.InternalServerError(
        "vendor said no", response=_response(503), body=None
    )


def _anthropic_503() -> Exception:
    import anthropic

    return anthropic.InternalServerError(
        "vendor said no", response=_response(503), body=None
    )


def _openai_timeout() -> Exception:
    import openai

    return openai.APITimeoutError(request=_REQUEST)


def _anthropic_timeout() -> Exception:
    import anthropic

    return anthropic.APITimeoutError(request=_REQUEST)


def _openai_provider() -> Any:
    from agent_platform.integrations.llm.openai.provider import OpenAILLM

    return OpenAILLM(OpenAICredentials(api_key=SecretStr("sk-test")), model="gpt-4.1")


def _anthropic_provider() -> Any:
    from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM

    return AnthropicLLM(
        AnthropicCredentials(api_key=SecretStr("sk-ant-test")),
        model="claude-sonnet-4-6",
    )


@dataclass(frozen=True)
class Vendor:
    patch_client: Callable[[Any], MagicMock]
    provider: Callable[[], Any]
    create: Callable[[MagicMock], AsyncMock]
    status: Callable[[MagicMock], AsyncMock]
    results: Callable[[MagicMock], AsyncMock]
    error_503: Callable[[], Exception]
    error_timeout: Callable[[], Exception]


def _patch_openai(mocker: Any) -> MagicMock:
    client_class = mocker.patch(
        "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
    )
    client = MagicMock()
    client_class.return_value = client
    client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_1"))
    return client


def _patch_anthropic(mocker: Any) -> MagicMock:
    client_class = mocker.patch(
        "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
    )
    client = MagicMock()
    client_class.return_value = client
    return client


VENDORS = {
    "openai": Vendor(
        _patch_openai,
        _openai_provider,
        lambda client: client.batches.create,
        lambda client: client.batches.retrieve,
        lambda client: client.batches.retrieve,
        _openai_503,
        _openai_timeout,
    ),
    "anthropic": Vendor(
        _patch_anthropic,
        _anthropic_provider,
        lambda client: client.messages.batches.create,
        lambda client: client.messages.batches.retrieve,
        lambda client: client.messages.batches.results,
        _anthropic_503,
        _anthropic_timeout,
    ),
}

_vendor = pytest.mark.parametrize(
    "vendor",
    [
        pytest.param(
            vendor,
            id=name,
            marks=pytest.mark.skipif(
                importlib.util.find_spec(name) is None,
                reason=f"{name} is not installed",
            ),
        )
        for name, vendor in VENDORS.items()
    ],
)


@_vendor
async def test_a_timeout_on_batch_creation_costs_one_call(
    vendor: Vendor, mocker, no_retry_sleep
):
    client = vendor.patch_client(mocker)
    create = vendor.create(client)
    create.side_effect = vendor.error_timeout()

    with pytest.raises(ProviderError) as caught:
        await vendor.provider().submit_batch(_REQUESTS)

    assert create.call_count == 1
    assert str(caught.value).startswith("Batch submission failed")


@_vendor
async def test_a_503_on_batch_creation_costs_one_call(
    vendor: Vendor, mocker, no_retry_sleep
):
    client = vendor.patch_client(mocker)
    create = vendor.create(client)
    create.side_effect = vendor.error_503()

    with pytest.raises(ProviderError):
        await vendor.provider().submit_batch(_REQUESTS)

    assert create.call_count == 1


@_vendor
async def test_a_503_on_a_status_read_costs_three_calls(
    vendor: Vendor, mocker, no_retry_sleep
):
    client = vendor.patch_client(mocker)
    status = vendor.status(client)
    status.side_effect = vendor.error_503()

    with pytest.raises(ProviderError) as caught:
        await vendor.provider().get_batch_status("batch_1")

    assert status.call_count == 3
    assert caught.value.retryable is True
    assert str(caught.value).startswith("Batch status check failed")


@_vendor
async def test_a_503_on_a_results_read_costs_three_calls(
    vendor: Vendor, mocker, no_retry_sleep
):
    client = vendor.patch_client(mocker)
    results = vendor.results(client)
    results.side_effect = vendor.error_503()

    with pytest.raises(ProviderError) as caught:
        await vendor.provider().fetch_batch_results("batch_1")

    assert results.call_count == 3
    assert caught.value.retryable is True
    assert str(caught.value).startswith("Batch result retrieval failed")
