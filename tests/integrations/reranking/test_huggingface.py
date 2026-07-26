from unittest.mock import MagicMock
from uuid import uuid4

from sentence_transformers import CrossEncoder

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking.huggingface.config import (
    HuggingFaceRerankerConfig,
)
from agent_platform.integrations.reranking.huggingface.provider import (
    HuggingFaceRerankerProvider,
)


def _mock_cross_encoder(mocker) -> MagicMock:
    mock_cls = mocker.patch(
        "agent_platform.integrations.reranking.huggingface.provider.CrossEncoder"
    )
    mock_cls.return_value = MagicMock(spec=CrossEncoder)
    return mock_cls


def _items() -> list[TextChunk]:
    return [
        TextChunk(id=uuid4(), text="a", index=0),
        TextChunk(id=uuid4(), text="b", index=1),
    ]


class TestSync:
    def test_device_forwarded_to_client(self, mocker):
        mock_cls = _mock_cross_encoder(mocker)
        provider = HuggingFaceRerankerProvider()
        provider._client(HuggingFaceRerankerConfig(device="cuda:0"))

        args, kwargs = mock_cls.call_args
        assert kwargs["device"] == "cuda:0"

    def test_scores_reranked_descending(self, mocker):
        mock_cls = _mock_cross_encoder(mocker)
        mock_cls.return_value.predict.return_value = [0.1, 0.9]

        provider = HuggingFaceRerankerProvider()
        results = provider.rerank("q", _items())

        assert [c.text for c in results] == ["b", "a"]

    def test_empty_items_short_circuits(self, mocker):
        mock_cls = _mock_cross_encoder(mocker)
        provider = HuggingFaceRerankerProvider()

        assert provider.rerank("q", []) == []
        mock_cls.assert_not_called()

    def test_return_scores_attaches_confidence(self, mocker):
        mock_cls = _mock_cross_encoder(mocker)
        mock_cls.return_value.predict.return_value = [0.2, 0.8]

        provider = HuggingFaceRerankerProvider()
        config = HuggingFaceRerankerConfig(return_scores=True, normalize_scores=True)
        results = provider.rerank("q", _items(), config)

        assert results[0].confidence.value == 1.0
        assert results[1].confidence.value == 0.0

    def test_top_k_applied(self, mocker):
        mock_cls = _mock_cross_encoder(mocker)
        mock_cls.return_value.predict.return_value = [0.1, 0.9]

        provider = HuggingFaceRerankerProvider()
        results = provider.rerank("q", _items(), HuggingFaceRerankerConfig(top_k=1))

        assert len(results) == 1
        assert results[0].text == "b"


class TestAsync:
    async def test_arerank_delegates_to_sync(self, mocker):
        mock_cls = _mock_cross_encoder(mocker)
        mock_cls.return_value.predict.return_value = [0.1, 0.9]

        provider = HuggingFaceRerankerProvider()
        results = await provider.arerank("q", _items())

        assert [c.text for c in results] == ["b", "a"]
