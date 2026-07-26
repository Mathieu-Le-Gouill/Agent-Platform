from unittest.mock import MagicMock
from uuid import uuid4

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking.flashrank.config import FlashRankConfig
from agent_platform.integrations.reranking.flashrank.provider import FlashRankReranker


def _items():
    return [
        TextChunk(id=uuid4(), text="a", index=0),
        TextChunk(id=uuid4(), text="b", index=1),
    ]


class TestSync:
    def test_max_length_forwarded_to_ranker(self, monkeypatch):
        captured = {}

        class FakeRanker:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def rerank(self, request):
                return [{"id": 0, "text": "a", "score": 0.5}]

        monkeypatch.setattr(
            "agent_platform.integrations.reranking.flashrank.provider.Ranker",
            FakeRanker,
        )

        reranker = FlashRankReranker()
        reranker.rerank("q", [_items()[0]], FlashRankConfig(max_length=256))

        assert captured["max_length"] == 256

    def test_max_length_omitted_when_unset(self, monkeypatch):
        captured = {}

        class FakeRanker:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def rerank(self, request):
                return [{"id": 0, "text": "a", "score": 0.5}]

        monkeypatch.setattr(
            "agent_platform.integrations.reranking.flashrank.provider.Ranker",
            FakeRanker,
        )

        reranker = FlashRankReranker()
        reranker.rerank("q", [_items()[0]], FlashRankConfig())

        assert "max_length" not in captured

    def test_return_scores_attaches_confidence(self):
        items = _items()
        reranker = FlashRankReranker()
        reranker._ranker = MagicMock()
        reranker._ranker.rerank.return_value = [
            {"id": 1, "text": "b", "score": 0.9},
            {"id": 0, "text": "a", "score": 0.1},
        ]

        config = FlashRankConfig(return_scores=True, normalize_scores=True)
        result = reranker.rerank("q", items, config)

        assert result[0].text == "b"
        assert result[0].confidence.value == 1.0
        assert result[1].text == "a"
        assert result[1].confidence.value == 0.0

    def test_return_scores_false_leaves_confidence_none(self):
        items = _items()
        reranker = FlashRankReranker()
        reranker._ranker = MagicMock()
        reranker._ranker.rerank.return_value = [
            {"id": 0, "text": "a", "score": 0.5},
            {"id": 1, "text": "b", "score": 0.7},
        ]

        result = reranker.rerank("q", items, FlashRankConfig())

        assert all(r.confidence is None for r in result)

    def test_top_k_applied_before_scoring(self):
        items = _items()
        reranker = FlashRankReranker()
        reranker._ranker = MagicMock()
        reranker._ranker.rerank.return_value = [
            {"id": 1, "text": "b", "score": 0.9},
            {"id": 0, "text": "a", "score": 0.1},
        ]

        config = FlashRankConfig(top_k=1, return_scores=True)
        result = reranker.rerank("q", items, config)

        assert len(result) == 1
        assert result[0].text == "b"

    def test_empty_items_short_circuits(self):
        reranker = FlashRankReranker()
        assert reranker.rerank("q", []) == []


class TestAsync:
    async def test_arerank_delegates_to_sync(self):
        items = _items()
        reranker = FlashRankReranker()
        reranker._ranker = MagicMock()
        reranker._ranker.rerank.return_value = [
            {"id": 0, "text": "a", "score": 0.5},
        ]

        result = await reranker.arerank("q", [items[0]])

        assert result[0].text == "a"
