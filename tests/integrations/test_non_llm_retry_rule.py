"""The retry rule of every remote embeddings, moderation, reranking, OCR and
speech-to-text provider, seen through its public method.

A 503 or a connection error costs 3 calls, a 400 costs 1 and raises a
non-retryable `ProviderError` that names the operation. Each case raises its
own SDK's real exception classes from the mocked SDK call.

Every SDK is optional, so each case imports its own lazily and is skipped when
one of the SDKs it needs is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
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
    HuggingFaceCredentials,
    JinaCredentials,
    MistralCredentials,
    OllamaCredentials,
    OpenAICredentials,
    VoyageCredentials,
)
from tests.helpers.retry_rule import (
    REQUEST,
    Case,
    Patch,
    google_api_status,
    http_response,
    httpx_connection,
    raised_from,
    requests_connection,
    retry_rule_tests,
    skip_unless_installed,
)

_EMBEDDING = "Embedding generation failed"
_MODERATION = "Moderation request failed"
_RERANKING = "Reranking failed"
_OCR = "OCR extraction failed"
_STT = "Speech-to-text failed"

_PROVIDER_MODULE = "agent_platform.integrations"


def _chunks() -> list[TextChunk]:
    return [TextChunk(id=uuid4(), text="a", index=0)]


def _audio() -> AudioChunk:
    return AudioChunk(data=b"audio", start=0, end=1000, format=AudioFormat.WAV)


def _document(tmp_path: Path) -> str:
    path = tmp_path / "scan.png"
    path.write_bytes(b"not-really-a-png")
    return str(path)


def _openai_status(status: int) -> Exception:
    import openai

    error_class = {
        400: openai.BadRequestError,
        503: openai.InternalServerError,
    }[status]
    return error_class("vendor said no", response=http_response(status), body=None)


def _openai_connection() -> Exception:
    import openai

    # The SDK raises this `from` the httpx transport error.
    return raised_from(
        openai.APIConnectionError(request=REQUEST),
        httpx.ConnectError("connection refused", request=REQUEST),
    )


def _voyage_connection() -> Exception:
    import aiohttp
    import voyageai.error

    # The SDK raises this `from` the aiohttp client error.
    return raised_from(
        voyageai.error.APIConnectionError("Error communicating with Voyage"),
        aiohttp.ClientConnectionError("connection refused"),
    )


def _genai_status(status: int) -> Exception:
    from google.genai import errors as genai_errors

    error_class = {
        400: genai_errors.ClientError,
        503: genai_errors.ServerError,
    }[status]
    return error_class(
        status, {"error": {"message": "vendor said no"}}, http_response(status)
    )


def _mistral_status(status: int) -> Exception:
    from mistralai.models.mistralerror import MistralError

    return MistralError("vendor said no", http_response(status))


def _ollama_status(status: int) -> Exception:
    from ollama import ResponseError

    return ResponseError("vendor said no", status)


def _huggingface_status(status: int) -> Exception:
    from huggingface_hub.errors import HfHubHTTPError

    return HfHubHTTPError("vendor said no", response=http_response(status))


def _cohere_status(status: int) -> Exception:
    import cohere.errors

    error_class = {
        400: cohere.errors.BadRequestError,
        503: cohere.errors.ServiceUnavailableError,
    }[status]
    return error_class(body="vendor said no")


def _voyage_status(status: int) -> Exception:
    import voyageai.error

    error_class = {
        400: voyageai.error.InvalidRequestError,
        503: voyageai.error.ServiceUnavailableError,
    }[status]
    return error_class("vendor said no", http_status=status)


def _httpx_status(status: int) -> Exception:
    return httpx.HTTPStatusError(
        "vendor said no", request=REQUEST, response=http_response(status)
    )


def _botocore_error(status: int, code: str) -> Exception:
    import botocore.exceptions

    return botocore.exceptions.ClientError(
        {
            "Error": {"Code": code, "Message": "vendor said no"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "DetectDocumentText",
    )


def _botocore_status(status: int) -> Exception:
    code = {400: "InvalidParameterException", 503: "ServiceUnavailableException"}[
        status
    ]
    return _botocore_error(status, code)


def _botocore_connection() -> Exception:
    import botocore.exceptions

    return botocore.exceptions.EndpointConnectionError(endpoint_url="https://x.test")


def _deepgram_status(status: int) -> Exception:
    import deepgram.core.api_error

    return deepgram.core.api_error.ApiError(status_code=status, body="vendor said no")


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


def _openai_key() -> OpenAICredentials:
    return OpenAICredentials(api_key=SecretStr("sk-test"))


async def _embed_openai(_: Path) -> object:
    from agent_platform.integrations.embeddings.openai.provider import (
        OpenAIEmbeddingProvider,
    )

    return await OpenAIEmbeddingProvider(_openai_key()).aembed_query("hi")


async def _embed_google(_: Path) -> object:
    from agent_platform.integrations.embeddings.google.provider import (
        GoogleEmbeddingProvider,
    )

    provider = GoogleEmbeddingProvider(GoogleCredentials(api_key=SecretStr("g-key")))
    return await provider.aembed_query("hi")


async def _embed_mistral(_: Path) -> object:
    from agent_platform.integrations.embeddings.mistral.provider import (
        MistralEmbeddingProvider,
    )

    provider = MistralEmbeddingProvider(MistralCredentials(api_key=SecretStr("m-key")))
    return await provider.aembed_query("hi")


async def _embed_ollama(_: Path) -> object:
    from agent_platform.integrations.embeddings.ollama.provider import (
        OllamaEmbeddingProvider,
    )

    return await OllamaEmbeddingProvider(OllamaCredentials()).aembed_query("hi")


async def _embed_huggingface_hosted(_: Path) -> object:
    from agent_platform.integrations.embeddings.huggingface.config import (
        HuggingFaceEmbeddingConfig,
        HuggingFaceEmbeddingMode,
    )
    from agent_platform.integrations.embeddings.huggingface.provider import (
        HuggingFaceEmbeddingProvider,
    )

    provider = HuggingFaceEmbeddingProvider(
        HuggingFaceCredentials(api_key=SecretStr("hf-key"))
    )
    config = HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED)
    return await provider.aembed_query("hi", config)


async def _moderate_openai(_: Path) -> object:
    from agent_platform.integrations.moderation.openai.provider import OpenAIModeration

    return await OpenAIModeration(_openai_key()).amoderate("hi")


async def _rerank_cohere(_: Path) -> object:
    from agent_platform.integrations.reranking.cohere.provider import (
        CohereRerankerProvider,
    )

    provider = CohereRerankerProvider(CohereCredentials(api_key=SecretStr("c-key")))
    return await provider.arerank("q", _chunks())


async def _rerank_voyage(_: Path) -> object:
    from agent_platform.integrations.reranking.voyage.provider import (
        VoyageRerankerProvider,
    )

    provider = VoyageRerankerProvider(VoyageCredentials(api_key=SecretStr("v-key")))
    return await provider.arerank("q", _chunks())


async def _rerank_jina(_: Path) -> object:
    from agent_platform.integrations.reranking.jina.provider import (
        JinaRerankerProvider,
    )

    provider = JinaRerankerProvider(JinaCredentials(api_key=SecretStr("j-key")))
    return await provider.arerank("q", _chunks())


async def _ocr_textract(path: Path) -> object:
    from agent_platform.integrations.ocr.aws_textract.provider import AWSTextractOCR

    credentials = AWSTextractCredentials(
        aws_access_key_id="id", aws_secret_access_key=SecretStr("secret")
    )
    return await AWSTextractOCR(credentials).extract(_document(path))


async def _ocr_google_vision(path: Path) -> object:
    from agent_platform.integrations.ocr.google_vision.provider import GoogleVisionOCR

    return await GoogleVisionOCR(GoogleVisionCredentials()).extract(_document(path))


async def _ocr_mistral(path: Path) -> object:
    from agent_platform.integrations.ocr.mistral.provider import MistralOCR

    credentials = MistralCredentials(api_key=SecretStr("m-key"))
    return await MistralOCR(credentials).extract(_document(path))


async def _stt_whisper(_: Path) -> object:
    from agent_platform.integrations.speech_to_text.openai.provider import (
        OpenAIWhisperSTT,
    )

    return await OpenAIWhisperSTT(_openai_key()).transcribe(_audio())


async def _stt_deepgram(_: Path) -> object:
    from agent_platform.integrations.speech_to_text.deepgram.provider import DeepgramSTT

    credentials = DeepgramCredentials(api_key=SecretStr("d-key"))
    return await DeepgramSTT(credentials).transcribe(_audio())


CASES = {
    "openai-embeddings": Case(
        _EMBEDDING,
        ("openai",),
        _patch_attribute(
            "openai.resources.embeddings.AsyncEmbeddings.create", AsyncMock
        ),
        _embed_openai,
        _openai_status,
        _openai_connection,
    ),
    "google-embeddings": Case(
        _EMBEDDING,
        ("google.genai",),
        _patch_attribute("google.genai.models.AsyncModels.embed_content", AsyncMock),
        _embed_google,
        _genai_status,
        httpx_connection,
    ),
    "mistral-embeddings": Case(
        _EMBEDDING,
        ("mistralai",),
        _patch_attribute("mistralai.embeddings.Embeddings.create_async", AsyncMock),
        _embed_mistral,
        _mistral_status,
        httpx_connection,
    ),
    "ollama-embeddings": Case(
        _EMBEDDING,
        ("ollama",),
        _patch_attribute("ollama.AsyncClient.embed", AsyncMock),
        _embed_ollama,
        _ollama_status,
        # The ollama client turns a refused connection into this builtin.
        lambda: ConnectionError("Failed to connect to Ollama"),
    ),
    "huggingface-hosted-embeddings": Case(
        _EMBEDDING,
        ("huggingface_hub", "sentence_transformers"),
        _patch_attribute(
            "huggingface_hub.AsyncInferenceClient.feature_extraction", AsyncMock
        ),
        _embed_huggingface_hosted,
        _huggingface_status,
        httpx_connection,
    ),
    "openai-moderation": Case(
        _MODERATION,
        ("openai",),
        _patch_attribute(
            "openai.resources.moderations.AsyncModerations.create", AsyncMock
        ),
        _moderate_openai,
        _openai_status,
        _openai_connection,
    ),
    "cohere-reranking": Case(
        _RERANKING,
        ("cohere",),
        _patch_attribute("cohere.AsyncClient.rerank", AsyncMock),
        _rerank_cohere,
        _cohere_status,
        httpx_connection,
    ),
    "voyage-reranking": Case(
        _RERANKING,
        ("voyageai", "aiohttp"),
        _patch_attribute("voyageai.AsyncClient.rerank", AsyncMock),
        _rerank_voyage,
        _voyage_status,
        _voyage_connection,
    ),
    "jina-reranking": Case(
        _RERANKING,
        (),
        _patch_attribute("httpx.AsyncClient.post", AsyncMock),
        _rerank_jina,
        _httpx_status,
        httpx_connection,
    ),
    "textract-ocr": Case(
        _OCR,
        ("boto3", "botocore"),
        _patch_attribute("botocore.client.BaseClient._make_api_call", MagicMock),
        _ocr_textract,
        _botocore_status,
        _botocore_connection,
    ),
    "google-vision-ocr": Case(
        _OCR,
        ("google.cloud.vision", "google.api_core", "requests"),
        _patch_google_vision_client,
        _ocr_google_vision,
        google_api_status,
        requests_connection,
    ),
    "mistral-ocr": Case(
        _OCR,
        ("mistralai",),
        _patch_attribute("mistralai.ocr.Ocr.process", MagicMock),
        _ocr_mistral,
        _mistral_status,
        httpx_connection,
    ),
    "openai-whisper": Case(
        _STT,
        ("openai",),
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
        ("deepgram",),
        _patch_attribute(
            "deepgram.listen.v1.media.client.AsyncMediaClient.transcribe_file",
            AsyncMock,
        ),
        _stt_deepgram,
        _deepgram_status,
        httpx_connection,
    ),
}


TestRetryRule = retry_rule_tests(CASES)


@skip_unless_installed("boto3", "botocore")
@pytest.mark.parametrize(
    "code", ["ThrottlingException", "ProvisionedThroughputExceededException"]
)
async def test_textract_throttling_sent_as_a_400_is_retried(
    code, mocker, no_retry_sleep, tmp_path
):
    sdk_call = CASES["textract-ocr"].patch(mocker, _botocore_error(400, code))

    with pytest.raises(ProviderError) as caught:
        await _ocr_textract(tmp_path)

    assert sdk_call.call_count == 3
    assert caught.value.retryable is True
    assert str(caught.value).startswith(_OCR)


@skip_unless_installed("cohere")
async def test_cohere_turns_its_own_retries_off_on_every_call(
    mocker, no_retry_sleep, tmp_path
):
    sdk_call = CASES["cohere-reranking"].patch(mocker, _cohere_status(400))

    with pytest.raises(ProviderError):
        await _rerank_cohere(tmp_path)

    assert sdk_call.call_args.kwargs["request_options"] == {"max_retries": 0}


@skip_unless_installed("google.cloud.vision", "google.api_core", "requests")
async def test_google_vision_turns_its_own_retry_off_on_every_call(
    mocker, no_retry_sleep, tmp_path
):
    sdk_call = _patch_google_vision_client(mocker, google_api_status(400))

    with pytest.raises(ProviderError):
        await _ocr_google_vision(tmp_path)

    assert sdk_call.call_args.kwargs["retry"] is None
