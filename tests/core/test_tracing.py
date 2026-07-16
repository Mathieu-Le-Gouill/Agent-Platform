import pytest

from agent_platform.core.tracing import TracingBackend, TracingConfig


class TestTracingConfigFromEnv:
    def test_defaults_to_none(self, monkeypatch):
        monkeypatch.delenv("AGENT_PLATFORM_TRACING", raising=False)
        assert TracingConfig.from_env().backend == TracingBackend.NONE

    def test_parses_langsmith(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "langsmith")
        assert TracingConfig.from_env().backend == TracingBackend.LANGSMITH

    def test_parses_langfuse(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "LANGFUSE")
        assert TracingConfig.from_env().backend == TracingBackend.LANGFUSE

    def test_invalid_value_defaults_to_none(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "not-a-backend")
        assert TracingConfig.from_env().backend == TracingBackend.NONE


class TestGetLangchainCallbacks:
    def test_none_backend_returns_empty(self):
        from agent_platform.integrations.llm.langchain_base import (
            get_langchain_callbacks,
        )

        assert get_langchain_callbacks(TracingConfig(backend=TracingBackend.NONE)) == []

    def test_langsmith_backend_returns_empty(self):
        from agent_platform.integrations.llm.langchain_base import (
            get_langchain_callbacks,
        )

        assert (
            get_langchain_callbacks(TracingConfig(backend=TracingBackend.LANGSMITH))
            == []
        )

    def test_langfuse_backend_returns_handler(self, monkeypatch):
        from agent_platform.integrations.llm import langchain_base

        sentinel = object()

        class FakeCallbackHandler:
            def __new__(cls, *args, **kwargs):
                return sentinel

        fake_module = type(
            "module", (), {"CallbackHandler": FakeCallbackHandler}
        )()
        monkeypatch.setitem(
            __import__("sys").modules, "langfuse.callback", fake_module
        )

        result = langchain_base.get_langchain_callbacks(
            TracingConfig(backend=TracingBackend.LANGFUSE)
        )
        assert result == [sentinel]
