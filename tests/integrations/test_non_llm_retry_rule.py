"""The retry rule of every remote embeddings, moderation, reranking, OCR and
speech-to-text provider, seen through its public method.

A 503 or a connection error costs 3 calls, a 400 costs 1 and raises a
non-retryable `ProviderError` that names the operation. Each case raises its
own SDK's real exception classes from the mocked SDK call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import aiohttp
import botocore.exceptions
import cohere.errors
import deepgram.core.api_error
import google.api_core.exceptions
import httpx
import openai
import pytest
import requests
import voyageai.error
from google.genai import errors as genai_errors
from mistralai.models.mistralerror import MistralError
from ollama import ResponseError
from pydantic import SecretStr

from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.chunk import AudioChunk, TextChunk
from agent_platform.core.schemas.enums import AudioFormat
from agent_platform.integrations.credentials import (
    AWSTextractCredentials,
    CohereCredentials,
    DeepgramCredentials,
    GoogleCredentials,
    GoogleVisionCredentials,
    JinaCredentials,
    MistralCredentials,
    OllamaCredentials,
    OpenAICredentials,
    VoyageCredentials,
)
from agent_platform.integrations.embeddings.google.provider import (
    GoogleEmbeddingProvider,
)
from agent_platform.integrations.embeddings.mistral.provider import (
    MistralEmbeddingProvider,
)
from agent_platform.integrations.embeddings.ollama.provider import (
    OllamaEmbeddingProvider,
)
from agent_platform.integrations.embeddings.openai.provider import (
    OpenAIEmbeddingProvider,
)
from agent_platform.integrations.moderation.openai.provider import OpenAIModeration
from agent_platform.integrations.ocr.aws_textract.provider import AWSTextractOCR
from agent_platform.integrations.ocr.google_vision.provider import GoogleVisionOCR
from agent_platform.integrations.ocr.mistral.provider import MistralOCR
from agent_platform.integrations.reranking.cohere.provider import (
    CohereRerankerProvider,
)
from agent_platform.integrations.reranking.jina.provider import JinaRerankerProvider
from agent_platform.integrations.reranking.voyage.provider import (
    VoyageRerankerProvider,
)
from agent_platform.integrations.speech_to_text.deepgram.provider import DeepgramSTT
from agent_platform.integrations.speech_to_text.openai.provider import (
    OpenAIWhisperSTT,
)

_REQUEST = httpx.Request("POST", "https://vendor.test/v1")

_EMBEDDING = "Embedding generation failed"
_MODERATION = "Moderation request failed"
_RERANKING = "Reranking failed"
_OCR = "OCR extraction failed"
_STT = "Speech-to-text failed"

_PROVIDER_MODULE = "agent_platform.integrations"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


def _chunks() -> list[TextChunk]:
    return [TextChunk(id=uuid4(), text="a", index=0)]


def _audio() -> AudioChunk:
    return AudioChunk(data=b"audio", start=0, end=1000, format=AudioFormat.WAV)


def _document(tmp_path: Path) -> str:
    path = tmp_path / "scan.png"
    path.write_bytes(b"not-really-a-png")
    return str(path)


def _openai_status(status: int) -> openai.APIStatusError:
    error_class = {
        400: openai.BadRequestError,
        503: openai.InternalServerError,
    }[status]
    return error_class("vendor said no", response=_response(status), body=None)


def _raised_from(error: Exception, cause: Exception) -> Exception:
    """Chain `error` to `cause` as `raise error from cause` does in an SDK."""
    error.__cause__ = cause
    return error


def _openai_connection() -> Exception:
    # The SDK raises this `from` the httpx transport error.
    return _raised_from(
        openai.APIConnectionError(request=_REQUEST),
        httpx.ConnectError("connection refused", request=_REQUEST),
    )


def _voyage_connection() -> Exception:
    # The SDK raises this `from` the aiohttp client error.
    return _raised_from(
        voyageai.error.APIConnectionError("Error communicating with Voyage"),
        aiohttp.ClientConnectionError("connection refused"),
    )


def _genai_status(status: int) -> Exception:
    error_class = {
        400: genai_errors.ClientError,
        503: genai_errors.ServerError,
    }[status]
    return error_class(
        status, {"error": {"message": "vendor said no"}}, _response(status)
    )


def _mistral_status(status: int) -> Exception:
    return MistralError("vendor said no", _response(status))


def _ollama_status(status: int) -> Exception:
    return ResponseError("vendor said no", status)


def _cohere_status(status: int) -> Exception:
    error_class = {
        400: cohere.errors.BadRequestError,
        503: cohere.errors.ServiceUnavailableError,
    }[status]
    return error_class(body="vendor said no")


def _voyage_status(status: int) -> Exception:
    error_class = {
        400: voyageai.error.InvalidRequestError,
        503: voyageai.error.ServiceUnavailableError,
    }[status]
    return error_class("vendor said no", http_status=status)


def _httpx_status(status: int) -> Exception:
    return httpx.HTTPStatusError(
        "vendor said no", request=_REQUEST, response=_response(status)
    )


def _httpx_connection() -> Exception:
    return httpx.ConnectError("connection refused", request=_REQUEST)


def _botocore_status(status: int) -> Exception:
    code = {400: "InvalidParameterException", 503: "ServiceUnavailableException"}[
        status
    ]
    return botocore.exceptions.ClientError(
        {
            "Error": {"Code": code, "Message": "vendor said no"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "DetectDocumentText",
    )


def _botocore_connection() -> Exception:
    return botocore.exceptions.EndpointConnectionError(endpoint_url="https://x.test")


def _google_api_status(status: int) -> Exception:
    error_class = {
        400: google.api_core.exceptions.InvalidArgument,
        503: google.api_core.exceptions.ServiceUnavailable,
    }[status]
    return error_class("vendor said no")


def _requests_connection() -> Exception:
    return requests.ConnectionError("connection refused")


def _deepgram_status(status: int) -> Exception:
    return deepgram.core.api_error.ApiError(status_code=status, body="vendor said no")


Patch = Callable[[Any, Exception], Any]
Call = Callable[[Path], Awaitable[object]]


def _patch_attribute(target: str, mock_class: type) -> Patch:
    """Replace `target` with a mock that raises, and return the mock."""

    def patch(mocker: Any, error: Exception) -> Any:
        mock = mock_class(side_effect=error)
        mocker.patch(target, new=mock)
        return mock

    return patch


def _patch_google_vision_client(mocker: Any, error: Exception) -> Any:
    client_class = mocker.patch(f"{_PROVIDER_MODULE}.ocr.google_vision.provider.vision")
    method = client_class.ImageAnnotatorClient.return_value.document_text_detection
    method.side_effect = error
    return method


@dataclass(frozen=True)
class Case:
    operation: str
    patch: Patch
    call: Call
    status_error: Callable[[int], Exception]
    connection_error: Callable[[], Exception]


def _openai_key() -> OpenAICredentials:
    return OpenAICredentials(api_key=SecretStr("sk-test"))


async def _embed_openai(_: Path) -> object:
    return await OpenAIEmbeddingProvider(_openai_key()).aembed_query("hi")


async def _embed_google(_: Path) -> object:
    provider = GoogleEmbeddingProvider(GoogleCredentials(api_key=SecretStr("g-key")))
    return await provider.aembed_query("hi")


async def _embed_mistral(_: Path) -> object:
    provider = MistralEmbeddingProvider(MistralCredentials(api_key=SecretStr("m-key")))
    return await provider.aembed_query("hi")


async def _embed_ollama(_: Path) -> object:
    return await OllamaEmbeddingProvider(OllamaCredentials()).aembed_query("hi")


async def _moderate_openai(_: Path) -> object:
    return await OpenAIModeration(_openai_key()).amoderate("hi")


async def _rerank_cohere(_: Path) -> object:
    provider = CohereRerankerProvider(CohereCredentials(api_key=SecretStr("c-key")))
    return await provider.arerank("q", _chunks())


async def _rerank_voyage(_: Path) -> object:
    provider = VoyageRerankerProvider(VoyageCredentials(api_key=SecretStr("v-key")))
    return await provider.arerank("q", _chunks())


async def _rerank_jina(_: Path) -> object:
    provider = JinaRerankerProvider(JinaCredentials(api_key=SecretStr("j-key")))
    return await provider.arerank("q", _chunks())


async def _ocr_textract(path: Path) -> object:
    credentials = AWSTextractCredentials(
        aws_access_key_id="id", aws_secret_access_key=SecretStr("secret")
    )
    return await AWSTextractOCR(credentials).extract(_document(path))


async def _ocr_google_vision(path: Path) -> object:
    return await GoogleVisionOCR(GoogleVisionCredentials()).extract(_document(path))


async def _ocr_mistral(path: Path) -> object:
    credentials = MistralCredentials(api_key=SecretStr("m-key"))
    return await MistralOCR(credentials).extract(_document(path))


async def _stt_whisper(_: Path) -> object:
    return await OpenAIWhisperSTT(_openai_key()).transcribe(_audio())


async def _stt_deepgram(_: Path) -> object:
    credentials = DeepgramCredentials(api_key=SecretStr("d-key"))
    return await DeepgramSTT(credentials).transcribe(_audio())


CASES = {
    "openai-embeddings": Case(
        _EMBEDDING,
        _patch_attribute(
            "openai.resources.embeddings.AsyncEmbeddings.create", AsyncMock
        ),
        _embed_openai,
        _openai_status,
        _openai_connection,
    ),
    "google-embeddings": Case(
        _EMBEDDING,
        _patch_attribute("google.genai.models.AsyncModels.embed_content", AsyncMock),
        _embed_google,
        _genai_status,
        _httpx_connection,
    ),
    "mistral-embeddings": Case(
        _EMBEDDING,
        _patch_attribute("mistralai.embeddings.Embeddings.create_async", AsyncMock),
        _embed_mistral,
        _mistral_status,
        _httpx_connection,
    ),
    "ollama-embeddings": Case(
        _EMBEDDING,
        _patch_attribute("ollama.AsyncClient.embed", AsyncMock),
        _embed_ollama,
        _ollama_status,
        # The ollama client turns a refused connection into this builtin.
        lambda: ConnectionError("Failed to connect to Ollama"),
    ),
    "openai-moderation": Case(
        _MODERATION,
        _patch_attribute(
            "openai.resources.moderations.AsyncModerations.create", AsyncMock
        ),
        _moderate_openai,
        _openai_status,
        _openai_connection,
    ),
    "cohere-reranking": Case(
        _RERANKING,
        _patch_attribute("cohere.AsyncClient.rerank", AsyncMock),
        _rerank_cohere,
        _cohere_status,
        _httpx_connection,
    ),
    "voyage-reranking": Case(
        _RERANKING,
        _patch_attribute("voyageai.AsyncClient.rerank", AsyncMock),
        _rerank_voyage,
        _voyage_status,
        _voyage_connection,
    ),
    "jina-reranking": Case(
        _RERANKING,
        _patch_attribute("httpx.AsyncClient.post", AsyncMock),
        _rerank_jina,
        _httpx_status,
        _httpx_connection,
    ),
    "textract-ocr": Case(
        _OCR,
        _patch_attribute("botocore.client.BaseClient._make_api_call", MagicMock),
        _ocr_textract,
        _botocore_status,
        _botocore_connection,
    ),
    "google-vision-ocr": Case(
        _OCR,
        _patch_google_vision_client,
        _ocr_google_vision,
        _google_api_status,
        _requests_connection,
    ),
    "mistral-ocr": Case(
        _OCR,
        _patch_attribute("mistralai.ocr.Ocr.process", MagicMock),
        _ocr_mistral,
        _mistral_status,
        _httpx_connection,
    ),
    "openai-whisper": Case(
        _STT,
        _patch_attribute(
            "openai.resources.audio.transcriptions.AsyncTranscriptions.create",
            AsyncMock,
        ),
        _stt_whisper,
        _openai_status,
        _openai_connection,
    ),
    "deepgram": Case(
        _STT,
        _patch_attribute(
            "deepgram.listen.v1.media.client.AsyncMediaClient.transcribe_file",
            AsyncMock,
        ),
        _stt_deepgram,
        _deepgram_status,
        _httpx_connection,
    ),
}

_case = pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())


@_case
async def test_a_503_costs_three_calls(case: Case, mocker, no_retry_sleep, tmp_path):
    sdk_call = case.patch(mocker, case.status_error(503))

    with pytest.raises(ProviderError) as caught:
        await case.call(tmp_path)

    assert sdk_call.call_count == 3
    assert caught.value.retryable is True
    assert str(caught.value).startswith(case.operation)


@_case
async def test_a_connection_error_costs_three_calls(
    case: Case, mocker, no_retry_sleep, tmp_path
):
    sdk_call = case.patch(mocker, case.connection_error())

    with pytest.raises(ProviderError) as caught:
        await case.call(tmp_path)

    assert sdk_call.call_count == 3
    assert caught.value.retryable is True
    assert str(caught.value).startswith(case.operation)


@_case
async def test_a_400_costs_one_call_and_is_not_retryable(
    case: Case, mocker, no_retry_sleep, tmp_path
):
    sdk_call = case.patch(mocker, case.status_error(400))

    with pytest.raises(ProviderError) as caught:
        await case.call(tmp_path)

    assert sdk_call.call_count == 1
    assert caught.value.retryable is False
    assert str(caught.value).startswith(case.operation)


async def test_cohere_turns_its_own_retries_off_on_every_call(
    mocker, no_retry_sleep, tmp_path
):
    sdk_call = CASES["cohere-reranking"].patch(mocker, _cohere_status(400))

    with pytest.raises(ProviderError):
        await _rerank_cohere(tmp_path)

    assert sdk_call.call_args.kwargs["request_options"] == {"max_retries": 0}


async def test_google_vision_turns_its_own_retry_off_on_every_call(
    mocker, no_retry_sleep, tmp_path
):
    sdk_call = _patch_google_vision_client(mocker, _google_api_status(400))

    with pytest.raises(ProviderError):
        await _ocr_google_vision(tmp_path)

    assert sdk_call.call_args.kwargs["retry"] is None
