from uuid import uuid4

from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import ScoreKind
from agent_platform.integrations.reranking.scoring import (
    apply_rerank_results,
    build_relevance_scores,
)


def test_normalize_min_max():
    scores = build_relevance_scores([0.2, 0.8], normalize=True)
    assert scores[0].value == 0.0
    assert scores[1].value == 1.0
    assert scores[0].kind == ScoreKind.RELEVANCE


def test_normalize_single_value_maps_to_one():
    scores = build_relevance_scores([0.5], normalize=True)
    assert scores[0].value == 1.0


def test_normalize_equal_values_map_to_one():
    scores = build_relevance_scores([0.5, 0.5], normalize=True)
    assert scores[0].value == 1.0
    assert scores[1].value == 1.0


def test_raw_out_of_range_values_do_not_raise():
    scores = build_relevance_scores([-3.2, 8.1], normalize=False)
    assert scores[0].value == -3.2
    assert scores[1].value == 8.1
    assert scores[0].low == float("-inf")
    assert scores[0].high == float("inf")


def test_none_values_preserved():
    scores = build_relevance_scores([0.5, None], normalize=True)
    assert scores[0] is not None
    assert scores[1] is None


def test_all_none_returns_all_none():
    scores = build_relevance_scores([None, None], normalize=True)
    assert scores == [None, None]


def _items():
    return [
        TextChunk(id=uuid4(), text="a", index=0),
        TextChunk(id=uuid4(), text="b", index=1),
    ]


def test_apply_rerank_results_maps_by_index():
    items = _items()
    results = [{"idx": 1, "score": 0.9}, {"idx": 0, "score": 0.1}]
    config = RerankerConfig()

    reranked = apply_rerank_results(
        items, results, config, index=lambda r: r["idx"], score=lambda r: r["score"]
    )

    assert [c.text for c in reranked] == ["b", "a"]


def test_apply_rerank_results_return_scores_false_leaves_confidence_none():
    items = _items()
    results = [{"idx": 0, "score": 0.9}, {"idx": 1, "score": 0.1}]
    config = RerankerConfig(return_scores=False)

    reranked = apply_rerank_results(
        items, results, config, index=lambda r: r["idx"], score=lambda r: r["score"]
    )

    assert all(c.confidence is None for c in reranked)


def test_apply_rerank_results_return_scores_true_attaches_confidence():
    items = _items()
    results = [{"idx": 0, "score": 0.2}, {"idx": 1, "score": 0.8}]
    config = RerankerConfig(return_scores=True, normalize_scores=True)

    reranked = apply_rerank_results(
        items, results, config, index=lambda r: r["idx"], score=lambda r: r["score"]
    )

    assert reranked[0].confidence.value == 0.0
    assert reranked[1].confidence.value == 1.0


def test_apply_rerank_results_applies_top_k():
    items = _items()
    results = [{"idx": 0, "score": 0.9}, {"idx": 1, "score": 0.1}]
    config = RerankerConfig(top_k=1)

    reranked = apply_rerank_results(
        items, results, config, index=lambda r: r["idx"], score=lambda r: r["score"]
    )

    assert len(reranked) == 1
    assert reranked[0].text == "a"
