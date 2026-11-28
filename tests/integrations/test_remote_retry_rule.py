"""The retry rule of every remote translation and vector store provider, seen
through its public method.

A 503 or a connection error costs 3 calls, a 400 costs 1 and raises a
non-retryable `ProviderError` that names the operation. Each case raises its
own SDK's real exception classes from the mocked SDK call. FAISS is local and
exempt.

Every SDK is optional, so each case imports its own lazily and is skipped when
one of the SDKs it needs is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import (
    AzureTranslatorCredentials,
    ChromaCredentials,
    DeepLCredentials,
    GoogleTranslateCredentials,
    PineconeCredentials,
    QdrantCredentials,
    WeaviateCredentials,
)
from tests.helpers.retry_rule import (
    Case,
    Patch,
    google_api_status,
    httpx_connection,
    raised_from,
    requests_connection,
    retry_rule_tests,
    skip_unless_installed,
)

_PROVIDER = "agent_platform.integrations"

_TRANSLATION = "Translation failed"
_SEARCH = "Vector store search failed"
_HYBRID_SEARCH = "Vector store hybrid search failed"

_TEXT = TextChunk(text="Hello", metadata={})
_VECTOR = [0.1, 0.2]


def _azure_status(status: int) -> Exception:
    import azure.core.exceptions

    response = MagicMock()
    response.status_code = status
    response.reason = "vendor said no"
    return azure.core.exceptions.HttpResponseError("vendor said no", response=response)


def _azure_connection() -> Exception:
    import azure.core.exceptions

    return azure.core.exceptions.ServiceRequestError("connection refused")


def _deepl_status(status: int) -> Exception:
    import deepl

    return deepl.DeepLException(
        "vendor said no", should_retry=status == 503, http_status_code=status
    )


def _deepl_connection() -> Exception:
    import deepl

    return deepl.ConnectionException("Connection failed", should_retry=True)


def _pinecone_status(status: int) -> Exception:
    import pinecone.exceptions

    error_class = {
        400: pinecone.exceptions.PineconeApiException,
        503: pinecone.exceptions.ServiceException,
    }[status]
    return error_class(status=status, reason="vendor said no")


def _aiohttp_connection() -> Exception:
    import aiohttp

    return aiohttp.ClientConnectionError("connection refused")


def _qdrant_status(status: int) -> Exception:
    from qdrant_client.http.exceptions import UnexpectedResponse

    return UnexpectedResponse(status, "vendor said no", b"", httpx.Headers())


def _qdrant_connection() -> Exception:
    from qdrant_client.http.exceptions import ResponseHandlingException

    # The client raises this `from` the httpx transport error.
    return raised_from(
        ResponseHandlingException(httpx_connection()), httpx_connection()
    )


def _qdrant_rate_limit() -> Exception:
    from qdrant_client.common.client_exceptions import ResourceExhaustedResponse

    # The client raises this, with no status, for a rate limit.
    return ResourceExhaustedResponse("rate limited", 5)


def _grpc_error(code_name: str) -> Exception:
    import grpc
    import grpc.aio

    return grpc.aio.AioRpcError(
        getattr(grpc.StatusCode, code_name), None, None, "vendor said no"
    )


def _weaviate_status(status: int) -> Exception:
    import weaviate.exceptions

    code_name = {400: "INVALID_ARGUMENT", 503: "UNAVAILABLE"}[status]
    # The client raises this `from` the gRPC error.
    return raised_from(
        weaviate.exceptions.WeaviateQueryError("vendor said no", "GRPC search"),
        _grpc_error(code_name),
    )


def _weaviate_connection() -> Exception:
    import weaviate.exceptions

    return weaviate.exceptions.WeaviateConnectionError("connection refused")


def _weaviate_timeout() -> Exception:
    import weaviate.exceptions

    return weaviate.exceptions.WeaviateTimeoutError("timed out")


def _weaviate_retry() -> Exception:
    import weaviate.exceptions

    return weaviate.exceptions.WeaviateRetryError("gave up", 3)


def _chroma_status(status: int) -> Exception:
    import chromadb.errors

    return {
        400: chromadb.errors.InvalidArgumentError,
        503: chromadb.errors.InternalError,
    }[status]("vendor said no")


def _patch_translate_method(patch_target: str, method: str) -> Patch:
    """Make `<patch_target>().<method>` raise, and return it."""

    def patch(mocker: Any, error: Exception) -> Any:
        client_class = mocker.patch(patch_target)
        sdk_call = MagicMock(side_effect=error)
        setattr(client_class.return_value, method, sdk_call)
        return sdk_call

    return patch


def _patch_google(mocker: Any, error: Exception) -> Any:
    module = mocker.patch(
        f"{_PROVIDER}.translation.google_translate.provider.google_translate"
    )
    module.Client.return_value.translate.side_effect = error
    return module.Client.return_value.translate


def _patch_pinecone(mocker: Any, error: Exception) -> Any:
    client_class = mocker.patch(
        f"{_PROVIDER}.vector_store.pinecone.provider.PineconeAsyncio"
    )
    index = MagicMock()
    index.__aenter__ = AsyncMock(return_value=index)
    index.__aexit__ = AsyncMock(return_value=False)
    index.query = AsyncMock(side_effect=error)
    client_class.return_value.IndexAsyncio.return_value = index
    return index.query


def _patch_qdrant_query(mocker: Any, error: Exception) -> Any:
    client_class = mocker.patch(
        f"{_PROVIDER}.vector_store.qdrant.provider.AsyncQdrantClient"
    )
    client_class.return_value.query_points = AsyncMock(side_effect=error)
    return client_class.return_value.query_points


def _patch_weaviate(mocker: Any, error: Exception) -> Any:
    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    near_vector = AsyncMock(side_effect=error)
    client.collections.get.return_value.query.near_vector = near_vector
    mocker.patch(
        f"{_PROVIDER}.vector_store.weaviate.provider.weaviate.use_async_with_local",
        return_value=client,
    )
    return near_vector


def _patch_chroma(mocker: Any, error: Exception) -> Any:
    import chromadb.api

    client = MagicMock(spec=chromadb.api.AsyncClientAPI)
    collection = MagicMock()
    collection.query = AsyncMock(side_effect=error)
    client.get_or_create_collection = AsyncMock(return_value=collection)
    mocker.patch("chromadb.AsyncHttpClient", AsyncMock(return_value=client))
    return collection.query


async def _translate_azure(_: Path) -> object:
    from agent_platform.integrations.translation.azure.provider import AzureTranslator

    translator = AzureTranslator(
        AzureTranslatorCredentials(api_key="test-key", region="westus")
    )
    return await translator.atranslate(_TEXT, target=Language.FR)


async def _translate_deepl(_: Path) -> object:
    from agent_platform.integrations.translation.deepl.provider import DeepLTranslator

    translator = DeepLTranslator(DeepLCredentials(auth_key="test-key"))
    return await translator.atranslate(_TEXT, target=Language.FR)


async def _translate_google(_: Path) -> object:
    from agent_platform.integrations.translation.google_translate.provider import (
        GoogleTranslator,
    )

    translator = GoogleTranslator(GoogleTranslateCredentials())
    return await translator.atranslate(_TEXT, target=Language.FR)


async def _search_pinecone(_: Path) -> object:
    from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
    from agent_platform.integrations.vector_store.pinecone.provider import (
        PineconeStore,
    )

    store = PineconeStore(PineconeCredentials(api_key="secret"))
    return await store.search_with_scores(
        _VECTOR, config=PineconeConfig(host="idx.svc.pinecone.io")
    )


async def _search_qdrant(_: Path) -> object:
    from agent_platform.integrations.vector_store.qdrant.provider import (
        QdrantVectorStoreProvider,
    )

    store = QdrantVectorStoreProvider(QdrantCredentials())
    return await store.search_with_scores(_VECTOR)


async def _hybrid_search_qdrant(_: Path) -> object:
    from agent_platform.core.schemas.vector import SparseVector
    from agent_platform.integrations.vector_store.qdrant.provider import (
        QdrantVectorStoreProvider,
    )

    store = QdrantVectorStoreProvider(QdrantCredentials())
    return await store.search_hybrid(
        _VECTOR, SparseVector(indices=[1], values=[1.0]), k=3
    )


async def _search_weaviate(_: Path) -> object:
    from agent_platform.integrations.vector_store.weaviate.provider import WeaviateStore

    store = WeaviateStore(WeaviateCredentials(api_key=None))
    return await store.search_with_scores(_VECTOR)


async def _plain_search_weaviate(_: Path) -> object:
    from agent_platform.integrations.vector_store.weaviate.provider import WeaviateStore

    store = WeaviateStore(WeaviateCredentials(api_key=None))
    return await store.search(_VECTOR)


async def _search_chroma(_: Path) -> object:
    from agent_platform.integrations.vector_store.chroma.config import ChromaConfig
    from agent_platform.integrations.vector_store.chroma.provider import ChromaStore

    store = ChromaStore(ChromaCredentials())
    return await store.search_with_scores(_VECTOR, config=ChromaConfig())


_AZURE = ("azure.core", "azure.ai.translation.text")
_QDRANT = ("qdrant_client",)
_WEAVIATE = ("weaviate", "grpc")

CASES = {
    "azure-translation": Case(
        _TRANSLATION,
        _AZURE,
        _patch_translate_method(
            f"{_PROVIDER}.translation.azure.provider.TextTranslationClient",
            "translate",
        ),
        _translate_azure,
        _azure_status,
        _azure_connection,
    ),
    "deepl-translation": Case(
        _TRANSLATION,
        ("deepl",),
        _patch_translate_method(
            f"{_PROVIDER}.translation.deepl.provider.deepl.Translator",
            "translate_text",
        ),
        _translate_deepl,
        _deepl_status,
        _deepl_connection,
    ),
    "google-translation": Case(
        _TRANSLATION,
        ("google.cloud.translate_v2", "google.api_core", "requests"),
        _patch_google,
        _translate_google,
        google_api_status,
        requests_connection,
    ),
    "pinecone-search": Case(
        _SEARCH,
        ("pinecone", "aiohttp"),
        _patch_pinecone,
        _search_pinecone,
        _pinecone_status,
        _aiohttp_connection,
    ),
    "qdrant-search": Case(
        _SEARCH,
        _QDRANT,
        _patch_qdrant_query,
        _search_qdrant,
        _qdrant_status,
        _qdrant_connection,
    ),
    "qdrant-hybrid-search": Case(
        _HYBRID_SEARCH,
        _QDRANT,
        _patch_qdrant_query,
        _hybrid_search_qdrant,
        _qdrant_status,
        _qdrant_connection,
    ),
    "weaviate-search": Case(
        _SEARCH,
        _WEAVIATE,
        _patch_weaviate,
        _plain_search_weaviate,
        _weaviate_status,
        _weaviate_connection,
    ),
    "weaviate-search-with-scores": Case(
        _SEARCH,
        _WEAVIATE,
        _patch_weaviate,
        _search_weaviate,
        _weaviate_status,
        _weaviate_connection,
    ),
    "chroma-search": Case(
        _SEARCH,
        ("chromadb",),
        _patch_chroma,
        _search_chroma,
        _chroma_status,
        httpx_connection,
    ),
}

TestRetryRule = retry_rule_tests(CASES)


class TestTransientErrorsWithoutAStatus:
    @skip_unless_installed(*_QDRANT)
    async def test_qdrant_rate_limit_costs_three_calls(
        self, mocker, no_retry_sleep, tmp_path
    ):
        sdk_call = _patch_qdrant_query(mocker, _qdrant_rate_limit())

        with pytest.raises(ProviderError) as caught:
            await _search_qdrant(tmp_path)

        assert sdk_call.call_count == 3
        assert caught.value.retryable is True
        assert str(caught.value).startswith(_SEARCH)

    @skip_unless_installed(*_WEAVIATE)
    @pytest.mark.parametrize(
        "make_error", [_weaviate_timeout, _weaviate_retry], ids=["timeout", "retry"]
    )
    async def test_weaviate_timeout_and_retry_errors_cost_three_calls(
        self, make_error, mocker, no_retry_sleep, tmp_path
    ):
        sdk_call = _patch_weaviate(mocker, make_error())

        with pytest.raises(ProviderError) as caught:
            await _plain_search_weaviate(tmp_path)

        assert sdk_call.call_count == 3
        assert caught.value.retryable is True
        assert str(caught.value).startswith(_SEARCH)


class TestTheSdkDoesNotRetry:
    @skip_unless_installed(*_AZURE)
    def test_azure_sets_retry_total_to_zero(self, mocker):
        from agent_platform.integrations.translation.azure.provider import (
            AzureTranslator,
        )

        client_class = mocker.patch(
            f"{_PROVIDER}.translation.azure.provider.TextTranslationClient"
        )
        translator = AzureTranslator(
            AzureTranslatorCredentials(api_key="test-key", region="westus")
        )

        translator._client(translator._default_config())

        assert client_class.call_args.kwargs["retry_total"] == 0

    @skip_unless_installed("deepl")
    def test_deepl_sets_its_process_wide_retry_count_to_zero(self):
        import deepl

        import agent_platform.integrations.translation.deepl.provider  # noqa: F401

        assert deepl.http_client.max_network_retries == 0

    @skip_unless_installed("pinecone", "aiohttp")
    async def test_pinecone_builds_every_client_with_a_single_attempt(self, mocker):
        from agent_platform.integrations.vector_store.pinecone.config import (
            PineconeConfig,
        )
        from agent_platform.integrations.vector_store.pinecone.provider import (
            PineconeStore,
        )

        attempts: list[int] = []

        class FakeJitterRetry:
            def __init__(self, attempts: int) -> None:
                attempts_seen.append(attempts)

        attempts_seen = attempts
        # `aiohttp_retry` is an optional extra of the SDK: stand in for its module.
        module = MagicMock(JitterRetry=FakeJitterRetry)
        mocker.patch.dict(
            "sys.modules", {"pinecone.openapi_support.retry_aiohttp": module}
        )
        client_class = mocker.patch(
            f"{_PROVIDER}.vector_store.pinecone.provider.PineconeAsyncio"
        )
        store = PineconeStore(PineconeCredentials(api_key="secret"))

        await store._index(PineconeConfig(host="idx.svc.pinecone.io"))

        client = client_class.return_value
        assert attempts == [1]
        assert isinstance(client._openapi_config.retries, FakeJitterRetry)
