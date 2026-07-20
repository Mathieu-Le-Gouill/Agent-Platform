from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agent_platform.core.schemas.chunk import TextChunk


@pytest.fixture
def openai_provider():
    from agent_platform.integrations.embeddings.openai.openai import (
        OpenAIEmbeddingProvider,
    )

    provider = OpenAIEmbeddingProvider()
    mock_lc = MagicMock()
    mock_lc.aembed_documents = AsyncMock()
    mock_lc.embed_documents = MagicMock()
    mock_lc.aembed_query = AsyncMock()
    mock_lc.embed_query = MagicMock()
    provider._client = MagicMock(return_value=mock_lc)
    return provider, mock_lc


@pytest.fixture
def mistral_provider():
    from agent_platform.integrations.embeddings.mistral.mistral import (
        MistralEmbeddingProvider,
    )

    provider = MistralEmbeddingProvider()
    mock_lc = MagicMock()
    mock_lc.aembed_documents = AsyncMock()
    mock_lc.embed_documents = MagicMock()
    mock_lc.aembed_query = AsyncMock()
    mock_lc.embed_query = MagicMock()
    provider._client = MagicMock(return_value=mock_lc)
    return provider, mock_lc


@pytest.fixture
def ollama_provider():
    from agent_platform.integrations.embeddings.ollama.ollama import (
        OllamaEmbeddingProvider,
    )

    provider = OllamaEmbeddingProvider()
    mock_lc = MagicMock()
    mock_lc.aembed_documents = AsyncMock()
    mock_lc.embed_documents = MagicMock()
    mock_lc.aembed_query = AsyncMock()
    mock_lc.embed_query = MagicMock()
    provider._client = MagicMock(return_value=mock_lc)
    return provider, mock_lc


@pytest.fixture
def two_text_chunks():
    uid1 = uuid4()
    uid2 = uuid4()
    return [
        TextChunk(id=uid1, text="hello"),
        TextChunk(id=uid2, text="world"),
    ], [uid1, uid2]
