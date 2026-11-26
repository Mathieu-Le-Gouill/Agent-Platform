from dataclasses import dataclass
from typing import Annotated

import pytest

from agent_platform.core.observability import (
    OPERATION_KEY,
    Fact,
    Observation,
    facts_of,
    flatten,
)


@dataclass
class Sample(Observation):
    operation = "sample"

    label: Annotated[str, Fact("sample.label")]
    detail: Annotated[str | None, Fact("sample.detail")] = None
    body: Annotated[object | None, Fact("sample.body", payload=True)] = None
    count: Annotated[int | None, Fact("sample.count")] = None
    untracked: str = "not a fact"


@dataclass
class Derived(Sample):
    extra: Annotated[str | None, Fact("sample.extra")] = None


@dataclass
class Dynamic(Observation):
    name: str

    @property
    def operation_name(self) -> str:
        return self.name


@pytest.fixture(autouse=True)
def content_capture_off(monkeypatch):
    monkeypatch.delenv("AGENT_PLATFORM_CAPTURE_CONTENT", raising=False)


class TestFactsOf:
    def test_finds_annotated_fields_only(self):
        assert [name for name, _ in facts_of(Sample)] == [
            "label",
            "detail",
            "body",
            "count",
        ]

    def test_includes_inherited_facts(self):
        assert dict(facts_of(Derived))["label"].key == "sample.label"
        assert "extra" in dict(facts_of(Derived))

    def test_is_cached_per_class(self):
        assert facts_of(Sample) is facts_of(Sample)


class TestFlatten:
    def test_records_the_operation_as_an_attribute(self):
        assert flatten(Sample(label="a"))[OPERATION_KEY] == "sample"

    def test_omits_the_operation_when_there_is_none(self):
        @dataclass
        class Event(Observation):
            value: Annotated[int, Fact("event.value")]

        assert OPERATION_KEY not in flatten(Event(value=1))

    def test_uses_a_dynamic_operation_name(self):
        assert flatten(Dynamic(name="embedder"))[OPERATION_KEY] == "embedder"

    def test_exports_facts_under_their_platform_keys(self):
        attributes = flatten(Sample(label="a", detail="d"))
        assert attributes["sample.label"] == "a"
        assert attributes["sample.detail"] == "d"

    def test_skips_unannotated_fields(self):
        assert "untracked" not in flatten(Sample(label="a"))

    def test_drops_fields_left_at_none(self):
        assert "sample.detail" not in flatten(Sample(label="a"))

    def test_a_payload_exports_only_its_size_without_capture(self):
        attributes = flatten(Sample(label="a", body={"q": "secret"}))
        assert attributes["sample.body.size"] > 0
        assert "sample.body.digest" not in attributes
        assert "sample.body.content" not in attributes

    def test_a_payload_exports_digest_and_content_when_capture_is_on(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_CAPTURE_CONTENT", "true")
        attributes = flatten(Sample(label="a", body="secret"))
        assert attributes["sample.body.size"] == 6
        assert attributes["sample.body.digest"]
        assert attributes["sample.body.content"] == "secret"


class TestObservation:
    def test_record_error_stores_the_exception(self):
        observation = Sample(label="a")
        error = ValueError("boom")
        observation.record_error(error)
        assert observation.error is error
