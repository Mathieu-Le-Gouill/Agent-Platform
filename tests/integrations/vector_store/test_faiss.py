import asyncio
import threading
from uuid import uuid4

import faiss
import numpy as np
import pytest

from agent_platform.core.errors import ConfigError, ValidationError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.vector_store.faiss.config import (
    DistanceMetric,
    FAISSConfig,
)
from agent_platform.integrations.vector_store.faiss.provider import FAISSStore


def _chunk(text: str, **kw) -> TextChunk:
    return TextChunk(id=uuid4(), text=text, index=0, **kw)


def _in_memory_ntotal(store: FAISSStore) -> int:
    return store._states[None].index.ntotal


def _files(directory) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


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
        assert store._states == {}

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

        assert _in_memory_ntotal(store) == 1

    async def test_non_json_metadata_is_rejected_before_any_change(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        before = _files(index_dir)

        with pytest.raises(ValidationError, match="JSON"):
            await store.add(
                [_chunk("bad", metadata={"x": object()})], [[0.0, 1.0]], config=config
            )

        assert _files(index_dir) == before
        results = await store.search(query_vector=[1.0, 1.0], k=5, config=config)
        assert [c.id for c in results] == [kept.id]

    async def test_duplicate_ids_in_one_batch_are_rejected_in_memory(self):
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]])
        dup = _chunk("dup")

        with pytest.raises(ValidationError, match=str(dup.id)):
            await store.add([dup, dup, _chunk("ok")], [[0.0, 1.0]] * 3)

        state = store._states[None]
        assert state.index.ntotal == len(state.docs) == 1
        assert dup.id not in state.uuid_to_id

    async def test_duplicate_ids_in_one_batch_are_rejected_when_persisted(
        self, tmp_path
    ):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        before = _files(index_dir)
        dup = _chunk("dup")

        with pytest.raises(ValidationError, match="repeated chunk ids"):
            await store.add([dup, dup], [[0.0, 1.0]] * 2, config=config)

        assert _files(index_dir) == before
        results = await FAISSStore().search([1.0, 1.0], k=5, config=config)
        assert [c.id for c in results] == [kept.id]

    async def test_duplicate_id_message_caps_the_list(self):
        chunks = [_chunk(f"c{i}") for i in range(7)]

        with pytest.raises(ValidationError) as exc_info:
            await FAISSStore().add(chunks * 2, [[1.0, 0.0]] * 14)

        message = str(exc_info.value)
        assert sum(str(c.id) in message for c in chunks) == 5
        assert "and 2 more" in message

    async def test_overwrite_across_batches_reloads_cleanly(self, tmp_path):
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        chunk = _chunk("v1")
        store = FAISSStore()
        await store.add([chunk], [[1.0, 0.0]], config=config)
        updated = chunk.model_copy(update={"text": "v2"})
        await store.add([updated], [[0.0, 1.0]], config=config)

        reloaded = FAISSStore()
        results = await reloaded.search([0.0, 1.0], k=5, config=config)
        state = next(iter(reloaded._states.values()))
        assert [c.text for c in results] == ["v2"]
        assert state.index.ntotal == len(state.docs) == 1

    async def test_in_memory_store_accepts_metadata_json_cannot_hold(self):
        store = FAISSStore()
        chunk = _chunk("numpy", metadata={"page": np.int64(3)})

        await store.add([chunk], [[1.0, 0.0]])

        assert [c.id for c in await store.search(query_vector=[1.0, 0.0])] == [chunk.id]


class TestFAISSDelete:
    async def test_delete_removes_from_index(self):
        store = FAISSStore()
        chunk = _chunk("hello")
        await store.add([chunk], [[1.0, 0.0]])

        await store.delete([chunk.id])

        assert await store.search(query_vector=[1.0, 0.0]) == []
        assert _in_memory_ntotal(store) == 0

    async def test_delete_unknown_id_is_noop(self):
        store = FAISSStore()
        await store.delete([uuid4()])

    async def test_delete_unknown_id_with_existing_index_is_noop(self):
        store = FAISSStore()
        chunk = _chunk("hello")
        await store.add([chunk], [[1.0, 0.0]])

        await store.delete([uuid4()])

        assert _in_memory_ntotal(store) == 1

    async def test_delete_with_no_index_is_noop(self):
        store = FAISSStore()
        await store.delete([uuid4()], config=FAISSConfig())


class TestFAISSRejectsNamespace:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s, c: s.add([_chunk("a")], [[1.0, 0.0]], config=c),
            lambda s, c: s.add([], [], config=c),
            lambda s, c: s.delete([uuid4()], config=c),
            lambda s, c: s.search(query_vector=[1.0, 0.0], config=c),
            lambda s, c: s.search_with_scores(query_vector=[1.0, 0.0], config=c),
        ],
        ids=["add", "add_empty", "delete", "search", "search_with_scores"],
    )
    async def test_operation_raises_and_touches_nothing(self, tmp_path, call):
        index_dir = tmp_path / "idx"
        store = FAISSStore()

        with pytest.raises(
            ConfigError, match="FAISS does not support namespace isolation"
        ):
            await call(
                store, FAISSConfig(index_path=str(index_dir), namespace="tenant-a")
            )

        assert store._states == {}
        assert not index_dir.exists()


class TestFAISSPerPathState:
    async def test_two_index_paths_on_one_instance_stay_separate(self, tmp_path):
        config_a = FAISSConfig(index_path=str(tmp_path / "a"))
        config_b = FAISSConfig(index_path=str(tmp_path / "b"))
        store = FAISSStore()
        chunk_a, chunk_b = _chunk("a"), _chunk("b")

        await store.add([chunk_a], [[1.0, 0.0]], config=config_a)
        await store.add([chunk_b], [[0.0, 1.0]], config=config_b)

        for config, expected in ((config_a, chunk_a.id), (config_b, chunk_b.id)):
            on_instance = await store.search([1.0, 1.0], k=5, config=config)
            on_disk = await FAISSStore().search([1.0, 1.0], k=5, config=config)
            assert [c.id for c in on_instance] == [expected]
            assert [c.id for c in on_disk] == [expected]

    async def test_in_memory_index_is_separate_from_persisted_ones(self, tmp_path):
        store = FAISSStore()
        persisted = FAISSConfig(index_path=str(tmp_path / "a"))
        in_memory, on_disk = _chunk("mem"), _chunk("disk")

        await store.add([in_memory], [[1.0, 0.0]])
        await store.add([on_disk], [[1.0, 0.0]], config=persisted)

        assert [c.id for c in await store.search([1.0, 0.0], k=5)] == [in_memory.id]
        assert [
            c.id for c in await store.search([1.0, 0.0], k=5, config=persisted)
        ] == [on_disk.id]

    async def test_same_directory_spelled_differently_shares_state(self, tmp_path):
        store = FAISSStore()
        first, second = _chunk("first"), _chunk("second")
        await store.add(
            [first], [[1.0, 0.0]], config=FAISSConfig(index_path=str(tmp_path / "idx"))
        )
        await store.add(
            [second],
            [[0.0, 1.0]],
            config=FAISSConfig(index_path=str(tmp_path / "x" / ".." / "idx")),
        )

        results = await FAISSStore().search(
            [1.0, 1.0], k=5, config=FAISSConfig(index_path=str(tmp_path / "idx"))
        )
        assert {c.id for c in results} == {first.id, second.id}


class TestFAISSConcurrency:
    async def test_concurrent_adds_keep_index_and_docstore_consistent(self, tmp_path):
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        store = FAISSStore()
        chunks = [_chunk(f"c{i}") for i in range(20)]

        await asyncio.gather(
            *(
                store.add([chunk], [[1.0, float(i)]], config=config)
                for i, chunk in enumerate(chunks)
            ),
            *(store.search([1.0, 0.0], k=3, config=config) for _ in range(5)),
        )

        reloaded = FAISSStore()
        results = await reloaded.search([1.0, 1.0], k=50, config=config)
        state = next(iter(reloaded._states.values()))
        assert {c.id for c in results} == {c.id for c in chunks}
        assert state.index.ntotal == len(state.docs) == 20
        assert len(set(state.docs)) == 20


class TestFAISSCancellation:
    @pytest.fixture
    def blocking_prepare(self, mocker):
        entered, release = threading.Event(), threading.Event()
        log: list[str] = []
        real_prepare = FAISSStore._prepare

        def prepare(self, vectors, distance):
            if vectors and vectors[0] == [9.0, 9.0]:
                entered.set()
                release.wait(5)
                log.append("blocked_done")
            elif vectors and vectors[0] == [0.0, 1.0]:
                log.append("follower")
            return real_prepare(self, vectors, distance)

        mocker.patch.object(FAISSStore, "_prepare", prepare)
        return entered, release, log

    @pytest.mark.parametrize("cancelled_op", ["add", "search"])
    async def test_cancelled_call_keeps_the_lock_until_its_thread_ends(
        self, tmp_path, blocking_prepare, cancelled_op
    ):
        entered, release, log = blocking_prepare
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        store = FAISSStore()
        first = _chunk("first")
        await store.add([first], [[1.0, 0.0]], config=config)
        blocked = _chunk("blocked")
        if cancelled_op == "add":
            op = store.add([blocked], [[9.0, 9.0]], config=config)
        else:
            op = store.search([9.0, 9.0], config=config)
        task = asyncio.ensure_future(op)
        await asyncio.to_thread(entered.wait, 5)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        follower = _chunk("follower")
        follow = asyncio.ensure_future(
            store.add([follower], [[0.0, 1.0]], config=config)
        )
        # A "has not run yet" check needs a time bound: there is no event for
        # a thread being parked on a lock. The log below makes the ordering
        # itself deterministic.
        await asyncio.sleep(0.1)
        assert not follow.done()

        release.set()
        await follow
        assert log == ["blocked_done", "follower"]

        expected = {first.id, follower.id}
        if cancelled_op == "add":
            expected.add(blocked.id)  # its thread ran to completion
        reloaded = FAISSStore()
        results = await reloaded.search([1.0, 1.0], k=10, config=config)
        state = next(iter(reloaded._states.values()))
        assert {c.id for c in results} == expected
        assert state.index.ntotal == len(state.docs) == len(expected)
