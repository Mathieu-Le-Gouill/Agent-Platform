from unittest.mock import AsyncMock

import pytest

from agent_platform.components.classification.embed_classifier.component import (
    EmbedClassifierInput,
    EmbeddingClassifier,
)
from agent_platform.components.classification.embed_classifier.config import (
    EmbeddingClassifierConfig,
)
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.chunking.recursive.config import RecursiveChunkerConfig


def _make_component(similarity_result=None):
    chunk = TextChunk(text="chunk")

    chunker = AsyncMock()
    chunker.arun = AsyncMock(return_value=[chunk])

    embedder = AsyncMock()
    embedder.arun = AsyncMock(
        side_effect=[
            EmbeddingResponse(
                embeddings=[Embedding.from_list([0.1, 0.2], model="m")], model="m"
            ),
            EmbeddingResponse(
                embeddings=[Embedding.from_list([0.3, 0.4], model="m")], model="m"
            ),
        ]
    )

    similarity_scorer = AsyncMock()
    similarity_scorer.arun = AsyncMock(
        return_value=similarity_result
        or ClassificationResult(
            predictions=[ClassificationPrediction(label="cat", score=Score(value=0.8))]
        )
    )

    component = EmbeddingClassifier(
        chunker=chunker,
        embedder=embedder,
        similarity_scorer=similarity_scorer,
    )
    return component, chunker, embedder, similarity_scorer


class TestEmbeddingClassifier:
    async def test_single_document_returns_similarity_result(self):
        component, chunker, embedder, similarity_scorer = _make_component()
        docs = [TextDocument(text="meow")]
        config = EmbeddingClassifierConfig()

        response = await component.arun(EmbedClassifierInput(docs, ["cat"], config))

        assert isinstance(response, ClassificationResponse)
        assert response.results[0].label == "cat"
        chunker.arun.assert_awaited_once_with(([docs[0]], None))
        assert embedder.arun.await_count == 2
        similarity_scorer.arun.assert_awaited_once()

    async def test_multiple_documents_produce_multiple_results(self):
        chunk1 = TextChunk(text="chunk1")
        chunk2 = TextChunk(text="chunk2")

        chunker = AsyncMock()
        chunker.arun = AsyncMock(side_effect=[[chunk1], [chunk2]])

        embedder = AsyncMock()
        embedder.arun = AsyncMock(
            side_effect=[
                EmbeddingResponse(
                    embeddings=[
                        Embedding.from_list([0.1, 0.2], model="m"),
                        Embedding.from_list([0.5, 0.6], model="m"),
                    ],
                    model="m",
                ),
                EmbeddingResponse(
                    embeddings=[Embedding.from_list([0.3, 0.4], model="m")], model="m"
                ),
            ]
        )

        similarity_scorer = AsyncMock()
        similarity_scorer.arun = AsyncMock(
            return_value=ClassificationResult(
                predictions=[
                    ClassificationPrediction(label="cat", score=Score(value=0.8))
                ]
            )
        )

        component = EmbeddingClassifier(
            chunker=chunker, embedder=embedder, similarity_scorer=similarity_scorer
        )
        docs = [TextDocument(text="a"), TextDocument(text="b")]
        response = await component.arun(
            EmbedClassifierInput(docs, ["cat"], EmbeddingClassifierConfig())
        )

        assert len(response.results) == 2
        assert chunker.arun.await_count == 2
        assert similarity_scorer.arun.await_count == 2

    async def test_forwards_explicit_chunker_and_embedding_configs(self):
        component, chunker, embedder, _ = _make_component()
        docs = [TextDocument(text="meow")]
        chunker_config = RecursiveChunkerConfig(chunk_size=128, chunk_overlap=0)
        embedding_config = EmbeddingConfig(batch_size=4)
        config = EmbeddingClassifierConfig(
            chunker_config=chunker_config, embedding_config=embedding_config
        )

        await component.arun(EmbedClassifierInput(docs, ["cat"], config))

        chunker.arun.assert_awaited_once_with(([docs[0]], chunker_config))
        assert [call.args[0].config for call in embedder.arun.await_args_list] == [
            embedding_config,
            embedding_config,
        ]

    async def test_unset_configs_leave_the_backend_defaults(self):
        component, chunker, embedder, _ = _make_component()
        docs = [TextDocument(text="meow")]

        await component.arun(
            EmbedClassifierInput(docs, ["cat"], EmbeddingClassifierConfig(top_k=2))
        )

        chunker.arun.assert_awaited_once_with(([docs[0]], None))
        assert [call.args[0].config for call in embedder.arun.await_args_list] == [
            None,
            None,
        ]

    async def test_embedding_count_mismatch_raises(self):
        component, chunker, embedder, _ = _make_component()
        chunker.arun = AsyncMock(
            return_value=[TextChunk(text="a"), TextChunk(text="b")]
        )

        with pytest.raises(ProviderError, match="1 embeddings for 2 chunks") as exc:
            await component.arun(
                EmbedClassifierInput(
                    [TextDocument(text="meow")], ["cat"], EmbeddingClassifierConfig()
                )
            )
        assert isinstance(exc.value.__cause__, ValueError)

    async def test_label_embedding_count_mismatch_raises(self):
        component, _, embedder, _ = _make_component()

        with pytest.raises(ProviderError, match="1 embeddings for 2 labels") as exc:
            await component.arun(
                EmbedClassifierInput(
                    [TextDocument(text="meow")],
                    ["cat", "dog"],
                    EmbeddingClassifierConfig(),
                )
            )
        assert isinstance(exc.value.__cause__, ValueError)
