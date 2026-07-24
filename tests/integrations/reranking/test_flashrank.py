from unittest.mock import MagicMock
from uuid import uuid4

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking.flashrank.config import FlashRankConfig
from agent_platform.integrations.reranking.flashrank.flashrank import FlashRankReranker


def _items():
    return [
        TextChunk(id=uuid4(), text="a", index=0),
        TextChunk(id=uuid4(), text="b", index=1),
    ]


async def test_max_length_forwarded_to_ranker(monkeypatch):
    captured = {}

    class FakeRanker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def rerank(self, request):
            return [{"id": 0, "text": "a", "score": 0.5}]

    monkeypatch.setattr(
        "agent_platform.integrations.reranking.flashrank.flashrank.Ranker", FakeRanker
    )

    reranker = FlashRankReranker()
    await reranker.rerank("q", [_items()[0]], FlashRankConfig(max_length=256))

    assert captured["max_length"] == 256


async def test_max_length_omitted_when_unset(monkeypatch):
    captured = {}

    class FakeRanker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def rerank(self, request):
            return [{"id": 0, "text": "a", "score": 0.5}]

    monkeypatch.setattr(
        "agent_platform.integrations.reranking.flashrank.flashrank.Ranker", FakeRanker
    )

    reranker = FlashRankReranker()
    await reranker.rerank("q", [_items()[0]], FlashRankConfig())

    assert "max_length" not in captured


async def test_return_scores_attaches_confidence():
    items = _items()
    reranker = FlashRankReranker()
    reranker._ranker = MagicMock()
    reranker._ranker.rerank.return_value = [
        {"id": 1, "text": "b", "score": 0.9},
        {"id": 0, "text": "a", "score": 0.1},
    ]

    config = FlashRankConfig(return_scores=True, normalize_scores=True)
    result = await reranker.rerank("q", items, config)

    assert result[0].text == "b"
    assert result[0].confidence.value == 1.0
    assert result[1].text == "a"
    assert result[1].confidence.value == 0.0


async def test_return_scores_false_leaves_confidence_none():
    items = _items()
    reranker = FlashRankReranker()
    reranker._ranker = MagicMock()
    reranker._ranker.rerank.return_value = [
        {"id": 0, "text": "a", "score": 0.5},
        {"id": 1, "text": "b", "score": 0.7},
    ]

    result = await reranker.rerank("q", items, FlashRankConfig())

    assert all(r.confidence is None for r in result)


async def test_top_k_applied_before_scoring():
    items = _items()
    reranker = FlashRankReranker()
    reranker._ranker = MagicMock()
    reranker._ranker.rerank.return_value = [
        {"id": 1, "text": "b", "score": 0.9},
        {"id": 0, "text": "a", "score": 0.1},
    ]

    config = FlashRankConfig(top_k=1, return_scores=True)
    result = await reranker.rerank("q", items, config)

    assert len(result) == 1
    assert result[0].text == "b"
