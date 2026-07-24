from agent_platform.core.schemas.score import ScoreKind
from agent_platform.integrations.reranking.scoring import build_relevance_scores


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
