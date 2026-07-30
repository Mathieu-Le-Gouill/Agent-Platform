from unittest.mock import AsyncMock

from agent_platform.components.embed_classifier.component import (
    EmbedClassifierInput,
    EmbeddingClassifier,
)
from agent_platform.components.embed_classifier.config import EmbeddingClassifierConfig
from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.core.schemas.score import Score


def _make_component(similarity_result=None, llm_classifier=None):
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
        llm_classifier=llm_classifier,
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

    async def test_llm_rerank_replaces_result_when_enabled(self):
        llm_classifier = AsyncMock()
        llm_classifier.arun = AsyncMock(
            return_value=ClassificationResponse(
                results=[
                    ClassificationResult(
                        predictions=[
                            ClassificationPrediction(
                                label="dog", score=Score(value=0.95)
                            )
                        ]
                    )
                ]
            )
        )
        component, *_ = _make_component(llm_classifier=llm_classifier)
        docs = [TextDocument(text="meow")]
        config = EmbeddingClassifierConfig(
            llm_rerank=True, llm_classifier_config=LLMClassifierConfig()
        )

        response = await component.arun(
            EmbedClassifierInput(docs, ["cat", "dog"], config)
        )

        assert response.results[0].label == "dog"
        llm_classifier.arun.assert_awaited_once()

    async def test_llm_rerank_skipped_when_no_llm_classifier_config(self):
        llm_classifier = AsyncMock()
        component, *_ = _make_component(llm_classifier=llm_classifier)
        docs = [TextDocument(text="meow")]
        config = EmbeddingClassifierConfig(llm_rerank=True, llm_classifier_config=None)

        response = await component.arun(EmbedClassifierInput(docs, ["cat"], config))

        assert response.results[0].label == "cat"
        llm_classifier.arun.assert_not_awaited()

    async def test_llm_rerank_ignored_when_llm_classifier_not_injected(self):
        component, *_ = _make_component(llm_classifier=None)
        docs = [TextDocument(text="meow")]
        config = EmbeddingClassifierConfig(
            llm_rerank=True, llm_classifier_config=LLMClassifierConfig()
        )

        response = await component.arun(EmbedClassifierInput(docs, ["cat"], config))

        assert response.results[0].label == "cat"

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
