from unittest.mock import MagicMock, patch

import pytest

from agent_platform.integrations.reranking.huggingface.huggingface import (
    HuggingFaceRerankerProvider,
    _ScoredCrossEncoderReranker,
)
from agent_platform.integrations.reranking.huggingface.config import (
    HuggingFaceRerankerConfig,
)


def _client(config: HuggingFaceRerankerConfig):
    with patch(
        "agent_platform.integrations.reranking.huggingface.huggingface.HuggingFaceCrossEncoder"
    ) as mock_encoder_cls:
        mock_encoder_cls.return_value = MagicMock()
        provider = HuggingFaceRerankerProvider()
        client = provider._client(config)
        return client, mock_encoder_cls


def test_device_routed_through_model_kwargs():
    _, mock_encoder_cls = _client(HuggingFaceRerankerConfig(device="cuda:0"))
    _, kwargs = mock_encoder_cls.call_args
    assert kwargs["model_kwargs"] == {"device": "cuda:0"}


def test_default_device_is_cpu():
    _, mock_encoder_cls = _client(HuggingFaceRerankerConfig())
    _, kwargs = mock_encoder_cls.call_args
    assert kwargs["model_kwargs"] == {"device": "cpu"}


def test_top_k_forwarded_as_top_n():
    client, _ = _client(HuggingFaceRerankerConfig(top_k=2))
    assert client.top_n == 2


def test_top_n_none_when_top_k_unset():
    client, _ = _client(HuggingFaceRerankerConfig())
    assert client.top_n is None


class TestScoredCrossEncoderReranker:
    def test_compress_documents_attaches_relevance_score(self):
        from langchain_core.documents import Document

        model = MagicMock()
        model.score.return_value = [0.1, 0.9]
        reranker = _ScoredCrossEncoderReranker(model=model)

        docs = [Document("a"), Document("b")]
        result = reranker.compress_documents(docs, "query")

        assert result[0].page_content == "b"
        assert result[0].metadata["relevance_score"] == 0.9
        assert result[1].page_content == "a"
        assert result[1].metadata["relevance_score"] == 0.1

    def test_compress_documents_respects_top_n(self):
        from langchain_core.documents import Document

        model = MagicMock()
        model.score.return_value = [0.1, 0.9, 0.5]
        reranker = _ScoredCrossEncoderReranker(model=model, top_n=1)

        docs = [Document("a"), Document("b"), Document("c")]
        result = reranker.compress_documents(docs, "query")

        assert len(result) == 1
        assert result[0].page_content == "b"

    def test_compress_documents_empty_input(self):
        model = MagicMock()
        reranker = _ScoredCrossEncoderReranker(model=model)
        assert reranker.compress_documents([], "query") == []

    async def test_acompress_documents_delegates_to_sync(self):
        from langchain_core.documents import Document

        model = MagicMock()
        model.score.return_value = [0.3]
        reranker = _ScoredCrossEncoderReranker(model=model)

        result = await reranker.acompress_documents([Document("a")], "query")
        assert result[0].metadata["relevance_score"] == 0.3
