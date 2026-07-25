import pytest

from agent_platform.evals.dataset import EvalDataset
from agent_platform.evals.errors import EvalDatasetError

_VALID_JSONL = (
    '{"id": "a", "input": "hi", "tags": ["x"]}\n'
    "\n"
    '{"id": "b", "input": "bye", "tags": ["y"]}\n'
)


class TestFromJsonl:
    def test_loads_cases(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text(_VALID_JSONL)

        dataset = EvalDataset.from_jsonl(path)

        assert len(dataset) == 2
        assert [c.id for c in dataset] == ["a", "b"]

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text("not json\n")

        with pytest.raises(EvalDatasetError):
            EvalDataset.from_jsonl(path)

    def test_invalid_schema_raises(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text('{"input": "missing id"}\n')

        with pytest.raises(EvalDatasetError):
            EvalDataset.from_jsonl(path)


class TestFilterByTag:
    def test_filters(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text(_VALID_JSONL)
        dataset = EvalDataset.from_jsonl(path)

        filtered = dataset.filter_by_tag("x")

        assert len(filtered) == 1
        assert next(iter(filtered)).id == "a"
