from uuid import uuid4

import pytest

from agent_platform.core.schemas.chunk import TextChunk
from tests.helpers import make_provider_with_mock_client

_ASYNC_METHODS = ("aembed_documents", "aembed_query")
_SYNC_METHODS = ("embed_documents", "embed_query")


@pytest.fixture
def openai_provider():
    from agent_platform.integrations.embeddings.openai.provider import (
        OpenAIEmbeddingProvider,
    )

    return make_provider_with_mock_client(
        OpenAIEmbeddingProvider,
        async_methods=_ASYNC_METHODS,
        sync_methods=_SYNC_METHODS,
    )


@pytest.fixture
def mistral_provider():
    from agent_platform.integrations.embeddings.mistral.provider import (
        MistralEmbeddingProvider,
    )

    return make_provider_with_mock_client(
        MistralEmbeddingProvider,
        async_methods=_ASYNC_METHODS,
        sync_methods=_SYNC_METHODS,
    )


@pytest.fixture
def ollama_provider():
    from agent_platform.integrations.embeddings.ollama.provider import (
        OllamaEmbeddingProvider,
    )

    return make_provider_with_mock_client(
        OllamaEmbeddingProvider,
        async_methods=_ASYNC_METHODS,
        sync_methods=_SYNC_METHODS,
    )


@pytest.fixture
def two_text_chunks():
    uid1 = uuid4()
    uid2 = uuid4()
    return [
        TextChunk(id=uid1, text="hello"),
        TextChunk(id=uid2, text="world"),
    ], [uid1, uid2]
