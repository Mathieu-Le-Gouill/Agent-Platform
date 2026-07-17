from agent_platform.utils.batching import chunked


def test_exact_multiple():
    result = list(chunked([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]


def test_remainder():
    result = list(chunked([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]


def test_empty_input():
    result = list(chunked([], 2))
    assert result == []


def test_size_larger_than_input():
    result = list(chunked([1, 2], 10))
    assert result == [[1, 2]]
