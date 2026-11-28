import hashlib
import json
import os
import pickle
import stat
from uuid import uuid4

import faiss
import pytest

from agent_platform.core.errors import ConfigError
from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.vector_store.faiss import provider as faiss_provider
from agent_platform.integrations.vector_store.faiss import storage as faiss_storage
from agent_platform.integrations.vector_store.faiss.config import (
    DistanceMetric,
    FAISSConfig,
)
from agent_platform.integrations.vector_store.faiss.provider import FAISSStore


def _chunk(text: str, **kw) -> TextChunk:
    return TextChunk(id=uuid4(), text=text, index=0, **kw)


def _files(directory) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def _index_name(directory) -> str:
    return json.loads((directory / "docstore.json").read_text())["index_file"]


def _edit_docstore(directory, **fields) -> None:
    path = directory / "docstore.json"
    payload = json.loads(path.read_text())
    payload.update(fields)
    path.write_text(json.dumps(payload))


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
        deserialize_index = mocker.spy(faiss_storage.faiss, "deserialize_index")

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

    @pytest.mark.parametrize("module", [faiss_provider, faiss_storage])
    def test_faiss_modules_do_not_import_pickle(self, module):
        assert not hasattr(module, "pickle")


class TestFAISSAtomicSave:
    async def test_failed_replace_leaves_previous_files_intact(self, tmp_path, mocker):
        index_dir = tmp_path / "idx"
        config = FAISSConfig(index_path=str(index_dir))
        store = FAISSStore()
        kept = _chunk("kept")
        await store.add([kept], [[1.0, 0.0]], config=config)
        before = _files(index_dir)
        mocker.patch.object(
            faiss_storage.os, "replace", side_effect=OSError("disk full")
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
            faiss_storage.faiss, "serialize_index", side_effect=RuntimeError("boom")
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

        mocker.patch.object(faiss_storage.os, "replace", side_effect=replace)
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
        real_remove = faiss_storage._remove_stray_files
        calls = {"n": 0}

        def remove_then_fail(directory, keep):
            calls["n"] += 1
            if calls["n"] == 2:  # the post-commit sweep
                raise PermissionError(13, "denied", "/private/dir/index-x.faiss")
            real_remove(directory, keep)

        mocker.patch.object(
            faiss_storage, "_remove_stray_files", side_effect=remove_then_fail
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
        mocker.patch.object(faiss_storage.os, "fsync", side_effect=OSError("io"))

        with pytest.raises(OSError, match="io"):
            await store.add([_chunk("lost")], [[0.0, 1.0]], config=config)
        mocker.stopall()

        assert _files(index_dir) == before
        results = await store.search([1.0, 1.0], k=5, config=config)
        assert [c.id for c in results] == [kept.id]
