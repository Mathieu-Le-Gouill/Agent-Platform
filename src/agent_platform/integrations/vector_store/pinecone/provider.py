from __future__ import annotations

import functools
from typing import Any
from uuid import UUID

from pinecone import PineconeAsyncio

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError, error_logged, require_secret
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import PineconeCredentials
from agent_platform.integrations.sdk_errors import translate_sdk_error
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
from agent_platform.integrations.vector_store.pinecone.mappers import (
    chunk_to_metadata,
    match_to_chunk,
)

_SEARCH_ERROR = "Vector store search failed"
_translate = functools.partial(translate_sdk_error, operation=_SEARCH_ERROR)


class PineconeStore(BaseVectorStore[PineconeConfig]):
    def __init__(
        self,
        credentials: PineconeCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, PineconeCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> PineconeConfig:
        return PineconeConfig()

    def _api_key(self) -> str:
        return require_secret(
            self._credentials.api_key, "Pinecone API key is required"
        ).get_secret_value()

    def _client_kwargs(self, config: PineconeConfig) -> dict[str, Any]:
        # `client_options.base_url` is intentionally not wired here:
        # `PineconeConfig.host` is a different, more specific concept (the
        # per-index data-plane host resolved via `describe_index`), not a
        # generic API base_url.
        kwargs: dict[str, Any] = {"api_key": self._api_key()}
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _new_client(self, config: PineconeConfig) -> PineconeAsyncio:
        pc = PineconeAsyncio(**self._client_kwargs(config))
        # The platform's `with_retry` is the only retry layer. The client
        # retries 5 times by default and only reads a `retries` option from
        # its OpenAPI configuration, which has no public setter. The option's
        # type comes from `aiohttp_retry`, which `pinecone[asyncio]` installs:
        # without it the client cannot send a request and raises its own error
        # that names the extra, so there is nothing to turn off.
        try:
            from pinecone.openapi_support.retry_aiohttp import JitterRetry
        except ImportError:
            return pc
        pc._openapi_config.retries = JitterRetry(attempts=1)
        return pc

    async def _resolve_host(self, config: PineconeConfig) -> str:
        if config.host is not None:
            return config.host
        async with self._new_client(config) as pc:
            index_model = await pc.describe_index(config.collection_name)
            if index_model.host is None:
                raise RuntimeError(
                    f"Pinecone index '{config.collection_name}' has no host"
                )
            return index_model.host

    async def _index(self, config: PineconeConfig) -> Any:
        host = await self._resolve_host(config)
        return self._new_client(config).IndexAsyncio(host=host)

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: PineconeConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        if not documents:
            return
        async with await self._index(config) as index:
            payload = [
                {
                    "id": str(doc.id),
                    "values": vector,
                    "metadata": chunk_to_metadata(doc),
                }
                for doc, vector in zip(documents, vectors)
            ]
            await index.upsert(vectors=payload, namespace=config.namespace)

    async def delete(
        self, document_ids: list[UUID], config: PineconeConfig | None = None
    ) -> None:
        config = config or self._default_config()
        if not document_ids:
            return
        async with await self._index(config) as index:
            await index.delete(
                ids=[str(doc_id) for doc_id in document_ids],
                namespace=config.namespace,
            )

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: PineconeConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        results = await self.search_with_scores(query_vector, k, config, filter)
        return [chunk for chunk, _ in results]

    @error_logged(re_raise=ProviderError, message=_SEARCH_ERROR)
    @with_retry(translate=_translate)
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: PineconeConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        async with await self._index(config) as index:
            response = await index.query(
                vector=query_vector,
                top_k=k,
                namespace=config.namespace,
                filter=filter or None,
                include_metadata=True,
            )
        return [
            (
                match_to_chunk(match),
                Score.similarity(min(1.0, max(0.0, match.score))),
            )
            for match in response.matches
        ]
