import hashlib
import json

import pytest

from agent_platform.core.observability import (
    capture_content,
    payload_facts,
)

PLATFORM_VAR = "AGENT_PLATFORM_CAPTURE_CONTENT"


@pytest.fixture(autouse=True)
def content_capture_off(monkeypatch):
    monkeypatch.delenv(PLATFORM_VAR, raising=False)


class TestCaptureContent:
    def test_off_by_default(self):
        assert capture_content() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_enabled_by_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(PLATFORM_VAR, value)
        assert capture_content() is True

    def test_disabled_by_anything_else(self, monkeypatch):
        monkeypatch.setenv(PLATFORM_VAR, "0")
        assert capture_content() is False

    def test_is_read_per_call_so_toggling_needs_no_restart(self, monkeypatch):
        monkeypatch.setenv(PLATFORM_VAR, "true")
        assert capture_content() is True
        monkeypatch.setenv(PLATFORM_VAR, "false")
        assert capture_content() is False


class TestPayloadFacts:
    def test_measures_a_string_as_is(self):
        assert payload_facts("hello").size == 5

    def test_omits_the_digest_unless_capture_is_on(self):
        # An unsalted hash of a small value space (an email, a yes/no) is
        # brute-forceable, so it is as sensitive as the content itself.
        assert payload_facts("hello").digest is None

    def test_includes_the_digest_when_capture_is_on(self, monkeypatch):
        monkeypatch.setenv(PLATFORM_VAR, "true")
        facts = payload_facts("hello")
        assert facts.digest == hashlib.sha256(b"hello").hexdigest()[:16]

    def test_serializes_anything_else_to_json(self):
        expected = json.dumps({"a": 1}, default=str, sort_keys=True)
        assert payload_facts({"a": 1}).size == len(expected.encode())

    def test_falls_back_to_str_for_unserializable_values(self):
        assert payload_facts({"a": object()}).size > 0

    def test_omits_content_unless_capture_is_on(self):
        assert payload_facts("secret").content is None

    def test_includes_content_when_capture_is_on(self, monkeypatch):
        monkeypatch.setenv(PLATFORM_VAR, "true")
        assert payload_facts("secret").content == "secret"
