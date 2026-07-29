from uuid import uuid4

import faiss
import pytest

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.vector_store.faiss.config import (
    DistanceMetric,
    FAISSConfig,
)
from agent_platform.integrations.vector_store.faiss.provider import FAISSStore


def _chunk(text: str, **kw) -> TextChunk:
    return TextChunk(id=uuid4(), text=text, index=0, **kw)


class TestFAISSDefaultConfig:
    def test_default_config(self):
        store = FAISSStore()
        assert isinstance(store._default_config(), FAISSConfig)


class TestFAISSNewIndex:
    def test_euclidean_uses_l2_metric(self):
        store = FAISSStore()
        index = store._new_index(4, DistanceMetric.EUCLIDEAN)
        assert index.index.metric_type == faiss.METRIC_L2

    def test_cosine_uses_inner_product_metric(self):
        store = FAISSStore()
        index = store._new_index(4, DistanceMetric.COSINE)
        assert index.index.metric_type == faiss.METRIC_INNER_PRODUCT

    def test_dot_uses_inner_product_metric(self):
        store = FAISSStore()
        index = store._new_index(4, DistanceMetric.DOT)
        assert index.index.metric_type == faiss.METRIC_INNER_PRODUCT


class TestFAISSPrepare:
    def test_cosine_normalizes(self):
        store = FAISSStore()
        result = store._prepare([[3.0, 4.0]], DistanceMetric.COSINE)
        assert result[0] == pytest.approx([0.6, 0.8])

    def test_euclidean_does_not_normalize(self):
        store = FAISSStore()
        result = store._prepare([[3.0, 4.0]], DistanceMetric.EUCLIDEAN)
        assert result[0] == pytest.approx([3.0, 4.0])

    def test_zero_vector_normalization_is_safe(self):
        store = FAISSStore()
        result = store._prepare([[0.0, 0.0]], DistanceMetric.COSINE)
        assert result[0] == pytest.approx([0.0, 0.0])


class TestFAISSToSimilarity:
    def test_euclidean_maps_zero_distance_to_one(self):
        store = FAISSStore()
        assert store._to_similarity(0.0, DistanceMetric.EUCLIDEAN) == 1.0

    def test_euclidean_clamped_to_zero_minimum(self):
        store = FAISSStore()
        assert store._to_similarity(1e9, DistanceMetric.EUCLIDEAN) >= 0.0

    def test_cosine_passes_through_and_clamps(self):
        store = FAISSStore()
        assert store._to_similarity(0.5, DistanceMetric.COSINE) == 0.5
        assert store._to_similarity(1.5, DistanceMetric.COSINE) == 1.0
        assert store._to_similarity(-0.5, DistanceMetric.COSINE) == 0.0


class TestFAISSAddAndSearch:
    async def test_add_creates_index_and_search_finds_it(self):
        store = FAISSStore()
        chunk = _chunk("hello")

        await store.add([chunk], [[1.0, 0.0, 0.0]], config=FAISSConfig())
        results = await store.search(query_vector=[1.0, 0.0, 0.0], k=1)

        assert len(results) == 1
        assert results[0].id == chunk.id
        assert results[0].text == "hello"

    async def test_search_with_scores_orders_by_similarity(self):
        store = FAISSStore()
        near = _chunk("near")
        far = _chunk("far")

        await store.add(
            [near, far],
            [[1.0, 0.0], [0.0, 1.0]],
            config=FAISSConfig(distance=DistanceMetric.COSINE),
        )
        results = await store.search_with_scores(query_vector=[1.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0][0].id == near.id
        assert results[0][1].value > results[1][1].value

    async def test_add_empty_list_is_noop(self):
        store = FAISSStore()
        await store.add([], [])
        assert store._index is None

    async def test_search_empty_index_returns_empty(self):
        store = FAISSStore()
        assert await store.search(query_vector=[0.1, 0.2]) == []
        assert await store.search_with_scores(query_vector=[0.1, 0.2]) == []

    async def test_search_respects_k(self):
        store = FAISSStore()
        chunks = [_chunk(f"c{i}") for i in range(5)]
        vectors = [[float(i), 0.0] for i in range(5)]
        await store.add(chunks, vectors)

        results = await store.search(query_vector=[0.0, 0.0], k=2)
        assert len(results) == 2

    async def test_search_filters_by_metadata(self):
        store = FAISSStore()
        match = _chunk("match", metadata={"source": "a.txt"})
        other = _chunk("other", metadata={"source": "b.txt"})
        await store.add([match, other], [[1.0, 0.0], [1.0, 0.01]])

        results = await store.search(
            query_vector=[1.0, 0.0], k=5, filter={"source": "a.txt"}
        )
        assert len(results) == 1
        assert results[0].id == match.id

    async def test_add_appends_to_existing_index(self):
        store = FAISSStore()
        first = _chunk("first")
        await store.add([first], [[1.0, 0.0]])

        second = _chunk("second")
        await store.add([second], [[0.0, 1.0]])

        results = await store.search(query_vector=[0.0, 1.0], k=2)
        assert {r.id for r in results} == {first.id, second.id}

    async def test_add_same_id_twice_updates_rather_than_duplicates(self):
        store = FAISSStore()
        chunk = _chunk("v1")
        await store.add([chunk], [[1.0, 0.0]])
        await store.add([chunk], [[1.0, 0.0]])

        assert store._index.ntotal == 1


class TestFAISSDelete:
    async def test_delete_removes_from_index(self):
        store = FAISSStore()
        chunk = _chunk("hello")
        await store.add([chunk], [[1.0, 0.0]])

        await store.delete([chunk.id])

        assert await store.search(query_vector=[1.0, 0.0]) == []
        assert store._index.ntotal == 0

    async def test_delete_unknown_id_is_noop(self):
        store = FAISSStore()
        await store.delete([uuid4()])

    async def test_delete_unknown_id_with_existing_index_is_noop(self):
        store = FAISSStore()
        chunk = _chunk("hello")
        await store.add([chunk], [[1.0, 0.0]])

        await store.delete([uuid4()])

        assert store._index.ntotal == 1

    async def test_delete_with_no_index_is_noop(self):
        store = FAISSStore()
        await store.delete([uuid4()], config=FAISSConfig())


class TestFAISSPersistence:
    async def test_save_and_load_round_trip(self, tmp_path):
        index_path = str(tmp_path / "idx")
        store = FAISSStore()
        chunk = _chunk("hello", metadata={"source": "a.txt"})
        await store.add(
            [chunk], [[1.0, 0.0]], config=FAISSConfig(index_path=index_path)
        )

        reloaded = FAISSStore()
        results = await reloaded.search(
            query_vector=[1.0, 0.0], config=FAISSConfig(index_path=index_path)
        )

        assert len(results) == 1
        assert results[0].id == chunk.id
        assert results[0].metadata["source"] == "a.txt"

    async def test_load_or_none_returns_none_without_index_path(self):
        store = FAISSStore()
        assert store._load_or_none(FAISSConfig()) is None

    async def test_load_or_none_returns_none_when_file_missing(self, tmp_path):
        store = FAISSStore()
        config = FAISSConfig(index_path=str(tmp_path / "nonexistent"))
        assert store._load_or_none(config) is None

    async def test_existing_in_memory_index_returned_without_reload(self, tmp_path):
        store = FAISSStore()
        chunk = _chunk("hello")
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        await store.add([chunk], [[1.0, 0.0]], config=config)

        existing = store._index
        assert store._load_or_none(config) is existing

    async def test_delete_persists_when_index_path_configured(self, tmp_path):
        index_path = str(tmp_path / "idx")
        config = FAISSConfig(index_path=index_path)
        store = FAISSStore()
        chunk = _chunk("hello")
        await store.add([chunk], [[1.0, 0.0]], config=config)
        await store.delete([chunk.id], config=config)

        reloaded = FAISSStore()
        results = await reloaded.search(query_vector=[1.0, 0.0], config=config)
        assert results == []
