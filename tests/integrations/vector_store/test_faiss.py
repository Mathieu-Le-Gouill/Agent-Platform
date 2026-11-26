import asyncio
import hashlib
import json
import os
import pickle
import stat
import threading
from uuid import uuid4

import faiss
import numpy as np
import pytest

from agent_platform.core.errors import ConfigError, ValidationError
from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.vector_store.faiss import provider as faiss_provider
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


def _index_name(directory) -> str:
    return json.loads((directory / "docstore.json").read_text())["index_file"]


def _edit_docstore(directory, **fields) -> None:
    path = directory / "docstore.json"
    payload = json.loads(path.read_text())
    payload.update(fields)
    path.write_text(json.dumps(payload))


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

    async def test_docstore_is_json_and_round_trips_every_field(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir), distance=DistanceMetric.DOT)
        chunk = TextChunk(
            text="hello",
            index=3,
            document_id=uuid4(),
            metadata={"source": "a.txt", "page": 2, "tags": ["x"]},
            start_char=1,
            end_char=6,
            confidence=Score.confidence(0.5),
            bbox=BoundingBox(x=1, y=2, width=3, height=4),
        )
        await FAISSStore().add([chunk], [[2.0, 0.0]], config=config)

        payload = json.loads((index_dir / "docstore.json").read_text())
        assert sorted(p.name for p in index_dir.iterdir()) == [
            "docstore.json",
            f"index-{payload['index_sha256']}.faiss",
        ]
        assert payload["index_file"] == f"index-{payload['index_sha256']}.faiss"
        assert payload["distance"] == "dot"
        assert payload["dimension"] == 2

        reloaded = FAISSStore()
        # the stored metric wins over the config default (COSINE would clamp to 1.0)
        results = await reloaded.search_with_scores(
            query_vector=[0.25, 0.0], config=FAISSConfig(index_path=str(index_dir))
        )
        assert results[0][0] == chunk
        assert results[0][1].value == pytest.approx(0.5)

    async def test_reloaded_store_keeps_assigning_fresh_ids(self, tmp_path):
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        first = _chunk("first")
        await FAISSStore().add([first], [[1.0, 0.0]], config=config)

        reloaded = FAISSStore()
        second = _chunk("second")
        await reloaded.add([second], [[0.0, 1.0]], config=config)

        results = await FAISSStore().search(query_vector=[1.0, 1.0], k=5, config=config)
        assert {r.id for r in results} == {first.id, second.id}

    async def test_search_returns_empty_when_directory_missing(self, tmp_path):
        store = FAISSStore()
        config = FAISSConfig(index_path=str(tmp_path / "nonexistent"))
        assert await store.search(query_vector=[1.0, 0.0], config=config) == []
        assert store._states == {}

    async def test_existing_in_memory_index_reused_without_reload(
        self, tmp_path, mocker
    ):
        store = FAISSStore()
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        await store.add([_chunk("hello")], [[1.0, 0.0]], config=config)
        deserialize_index = mocker.spy(faiss_provider.faiss, "deserialize_index")

        await store.search(query_vector=[1.0, 0.0], config=config)

        deserialize_index.assert_not_called()

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

    async def test_invalid_docstore_raises_config_error(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        (index_dir / "docstore.json").write_text("{not json")

        with pytest.raises(ConfigError, match="docstore"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    async def test_orphan_index_without_docstore_is_uncommitted(self, tmp_path):
        # what an interrupted first save leaves: nothing was committed
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        (index_dir / "docstore.json").unlink()

        store = FAISSStore()
        assert await store.search(query_vector=[1.0, 0.0], config=config) == []
        fresh = _chunk("fresh")
        await store.add([fresh], [[1.0, 0.0]], config=config)

        payload = json.loads((index_dir / "docstore.json").read_text())
        assert sorted(p.name for p in index_dir.iterdir()) == [
            "docstore.json",
            payload["index_file"],
        ]
        results = await FAISSStore().search([1.0, 0.0], k=5, config=config)
        assert [c.id for c in results] == [fresh.id]

    async def test_docstore_without_index_raises_config_error(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        (index_dir / _index_name(index_dir)).unlink()

        with pytest.raises(ConfigError, match="references a missing index-"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    async def test_index_altered_after_commit_raises_config_error(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        (index_dir / _index_name(index_dir)).write_bytes(b"tampered")

        with pytest.raises(ConfigError, match="does not match its docstore"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    @pytest.mark.parametrize(
        "index_file", ["../outside.faiss", "index.faiss", "index-abc.faiss"]
    )
    async def test_docstore_cannot_point_at_another_file(self, tmp_path, index_file):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        _edit_docstore(index_dir, index_file=index_file)

        with pytest.raises(ConfigError, match="unexpected index_file"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    async def test_unsupported_docstore_version_raises_config_error(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        _edit_docstore(index_dir, version=2)

        with pytest.raises(ConfigError, match="unsupported version 2"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    async def test_corrupt_index_raises_config_error(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        garbage = b"not a faiss index"
        sha256 = hashlib.sha256(garbage).hexdigest()
        (index_dir / f"index-{sha256}.faiss").write_bytes(garbage)
        # re-pin the docstore so the hash check passes and parsing is reached
        _edit_docstore(
            index_dir, index_sha256=sha256, index_file=f"index-{sha256}.faiss"
        )

        with pytest.raises(ConfigError, match="cannot be read"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    @pytest.mark.parametrize(
        "edit",
        [{"ntotal": 5}, {"next_id": 0}],
        ids=["ntotal_mismatch", "id_not_below_next_id"],
    )
    async def test_inconsistent_docstore_raises_config_error(self, tmp_path, edit):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        await FAISSStore().add([_chunk("hello")], [[1.0, 0.0]], config=config)
        _edit_docstore(index_dir, **edit)

        with pytest.raises(ConfigError, match="inconsistent"):
            await FAISSStore().search(query_vector=[1.0, 0.0], config=config)

    @pytest.mark.parametrize("umask", [0o022, 0o077])
    async def test_saved_files_follow_the_umask(self, tmp_path, umask):
        index_dir = tmp_path / "idx"
        previous = os.umask(umask)
        try:
            await FAISSStore().add(
                [_chunk("hello")],
                [[1.0, 0.0]],
                config=FAISSConfig(index_path=str(index_dir)),
            )
        finally:
            os.umask(previous)

        for path in index_dir.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask

    async def test_round_trip_without_directory_fsync_support(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
        config = FAISSConfig(index_path=str(tmp_path / "idx"))
        chunk = _chunk("hello")

        await FAISSStore().add([chunk], [[1.0, 0.0]], config=config)

        results = await FAISSStore().search([1.0, 0.0], config=config)
        assert [c.id for c in results] == [chunk.id]


class TestFAISSLegacyPickleDocstore:
    @pytest.fixture
    def legacy_dir(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        faiss.write_index(
            FAISSStore()._new_index(2, DistanceMetric.COSINE),
            str(index_dir / "index.faiss"),
        )
        (index_dir / "docstore.pkl").write_bytes(pickle.dumps({"docs": {}}))
        return index_dir

    @pytest.mark.parametrize(
        "call",
        [
            lambda s, c: s.add([_chunk("a")], [[1.0, 0.0]], config=c),
            lambda s, c: s.delete([uuid4()], config=c),
            lambda s, c: s.search(query_vector=[1.0, 0.0], config=c),
        ],
        ids=["add", "delete", "search"],
    )
    async def test_legacy_docstore_raises_and_is_never_unpickled(
        self, legacy_dir, mocker, call
    ):
        load = mocker.patch("pickle.load")
        loads = mocker.patch("pickle.loads")
        before = _files(legacy_dir)

        with pytest.raises(ConfigError, match="legacy pickle docstore") as exc_info:
            await call(FAISSStore(), FAISSConfig(index_path=str(legacy_dir)))

        assert "re-ingest" in str(exc_info.value)
        load.assert_not_called()
        loads.assert_not_called()
        assert _files(legacy_dir) == before

    def test_provider_does_not_import_pickle(self):
        assert not hasattr(faiss_provider, "pickle")


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


class TestFAISSAtomicSave:
    async def test_failed_replace_leaves_previous_files_intact(self, tmp_path, mocker):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        before = _files(index_dir)
        mocker.patch.object(
            faiss_provider.os, "replace", side_effect=OSError("disk full")
        )

        with pytest.raises(OSError, match="disk full"):
            await store.add([_chunk("lost")], [[0.0, 1.0]], config=config)

        mocker.stopall()
        assert _files(index_dir) == before
        # the failed add is not served from memory either: state reloads from disk
        results = await store.search([1.0, 1.0], k=5, config=config)
        assert [c.id for c in results] == [kept.id]

    async def test_failed_write_leaves_previous_files_intact(self, tmp_path, mocker):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        before = _files(index_dir)
        mocker.patch.object(
            faiss_provider.faiss, "serialize_index", side_effect=RuntimeError("boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await store.delete([kept.id], config=config)

        mocker.stopall()
        assert _files(index_dir) == before
        results = await store.search([1.0, 0.0], k=5, config=config)
        assert [c.id for c in results] == [kept.id]

    @pytest.mark.parametrize(
        "failing_replace", [1, 2], ids=["index_write", "docstore_commit"]
    )
    async def test_failure_before_the_commit_keeps_the_previous_pair(
        self, tmp_path, mocker, failing_replace
    ):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        committed = _files(index_dir)
        real_replace = os.replace
        calls = {"n": 0}

        def replace(src, dst):
            calls["n"] += 1
            if calls["n"] == failing_replace:
                raise OSError("crash")
            real_replace(src, dst)

        mocker.patch.object(faiss_provider.os, "replace", side_effect=replace)
        with pytest.raises(OSError, match="crash"):
            await store.add([_chunk("new")], [[0.0, 1.0]], config=config)
        mocker.stopall()

        # the committed pair is untouched; at most one new orphan index exists
        after = _files(index_dir)
        assert {k: v for k, v in after.items() if k in committed} == committed
        assert len(set(after) - set(committed)) == failing_replace - 1
        for reader in (store, FAISSStore()):
            results = await reader.search([1.0, 1.0], k=5, config=config)
            assert [c.id for c in results] == [kept.id]

    async def test_crash_leftovers_are_ignored_then_cleaned(self, tmp_path):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        kept = _chunk("kept")
        await FAISSStore().add([kept], [[1.0, 0.0]], config=config)
        # what a killed process can leave: a temp file and an unreferenced index
        (index_dir / f".docstore.json.{'a' * 32}.tmp").write_bytes(b"partial")
        (index_dir / f".index-{'1' * 64}.faiss.{'b' * 32}.tmp").write_bytes(b"part")
        (index_dir / f"index-{'0' * 64}.faiss").write_bytes(b"orphan")

        store = FAISSStore()
        assert [c.id for c in await store.search([1.0, 0.0], config=config)] == [
            kept.id
        ]
        added = _chunk("added")
        await store.add([added], [[0.0, 1.0]], config=config)

        assert sorted(p.name for p in index_dir.iterdir()) == [
            "docstore.json",
            _index_name(index_dir),
        ]
        results = await FAISSStore().search([1.0, 1.0], k=5, config=config)
        assert {c.id for c in results} == {kept.id, added.id}

    async def test_files_from_other_tools_survive_a_save(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        others = {
            ".other.tmp": b"other tool",
            "notes.txt": b"notes",
            ".docstore.json.dead.tmp": b"not our naming",
            "index-mine.faiss": b"not content-addressed",
        }
        for name, data in others.items():
            (index_dir / name).write_bytes(data)
        config = FAISSConfig(index_path=str(index_dir))

        store = FAISSStore()
        await store.add([_chunk("a")], [[1.0, 0.0]], config=config)
        await store.add([_chunk("b")], [[0.0, 1.0]], config=config)

        for name, data in others.items():
            assert (index_dir / name).read_bytes() == data

    async def test_cleanup_failure_after_commit_does_not_fail_the_add(
        self, tmp_path, mocker, caplog
    ):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        await store.add([_chunk("first")], [[1.0, 0.0]], config=config)
        real_remove = faiss_provider._remove_stray_files
        calls = {"n": 0}

        def remove_then_fail(directory, keep):
            calls["n"] += 1
            if calls["n"] == 2:  # the post-commit sweep
                raise PermissionError(13, "denied", "/private/dir/index-x.faiss")
            real_remove(directory, keep)

        mocker.patch.object(
            faiss_provider, "_remove_stray_files", side_effect=remove_then_fail
        )
        second = _chunk("second")

        await store.add([second], [[0.0, 1.0]], config=config)

        assert calls["n"] == 2
        assert "index-x.faiss" in caplog.text
        assert "PermissionError" in caplog.text
        assert "/private/dir" not in caplog.text
        results = await FAISSStore().search([1.0, 1.0], k=5, config=config)
        assert second.id in {c.id for c in results}

    async def test_failed_fsync_leaves_files_unchanged_and_no_temp(
        self, tmp_path, mocker
    ):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        before = _files(index_dir)
        mocker.patch.object(faiss_provider.os, "fsync", side_effect=OSError("io"))

        with pytest.raises(OSError, match="io"):
            await store.add([_chunk("lost")], [[0.0, 1.0]], config=config)
        mocker.stopall()

        assert _files(index_dir) == before
        results = await store.search([1.0, 1.0], k=5, config=config)
        assert [c.id for c in results] == [kept.id]


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
