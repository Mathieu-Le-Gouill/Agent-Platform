from agent_platform.models.chunk import TextChunk
from agent_platform.utils.score import filter_by_score, sort_by_score


class TestFilterByScore:
    def make_chunks(self, confidences: list[float]) -> list[TextChunk]:
        return [
            TextChunk(
                text=f"chunk{i}",
                index=i,
                metadata={"confidence": c},
            )
            for i, c in enumerate(confidences)
        ]

    def test_zero_threshold_returns_all(self):
        chunks = self.make_chunks([0.1, 0.5, 0.9])
        key = lambda c: c.metadata.get("confidence") or 1.0
        result = filter_by_score(chunks, 0.0, key=key)
        assert len(result) == 3

    def test_filters_below_threshold(self):
        chunks = self.make_chunks([0.1, 0.5, 0.9])
        key = lambda c: c.metadata.get("confidence") or 1.0
        result = filter_by_score(chunks, 0.5, key=key)
        assert len(result) == 2
        assert result[0].text == "chunk1"
        assert result[1].text == "chunk2"

    def test_all_below_returns_empty(self):
        chunks = self.make_chunks([0.1, 0.2])
        key = lambda c: c.metadata.get("confidence") or 1.0
        result = filter_by_score(chunks, 0.5, key=key)
        assert result == []

    def test_missing_confidence_defaults_to_1(self):
        chunks = [TextChunk(text="no_conf", index=0, metadata={})]
        key = lambda c: c.metadata.get("confidence") or 1.0
        result = filter_by_score(chunks, 0.5, key=key)
        assert len(result) == 1

    def test_works_with_any_key(self):
        items = [("a", 10), ("b", 5), ("c", 8)]
        result = filter_by_score(items, 7, key=lambda x: x[1])
        assert result == [("a", 10), ("c", 8)]


class TestSortByScore:
    def test_sorts_descending_by_default(self):
        items = [("a", 10), ("b", 5), ("c", 8)]
        result = sort_by_score(items, key=lambda x: x[1])
        assert result == [("a", 10), ("c", 8), ("b", 5)]

    def test_sorts_ascending_when_reverse_false(self):
        items = [("a", 10), ("b", 5), ("c", 8)]
        result = sort_by_score(items, key=lambda x: x[1], reverse=False)
        assert result == [("b", 5), ("c", 8), ("a", 10)]

    def test_does_not_mutate_original(self):
        items = [("a", 10), ("b", 5)]
        sort_by_score(items, key=lambda x: x[1])
        assert items == [("a", 10), ("b", 5)]
