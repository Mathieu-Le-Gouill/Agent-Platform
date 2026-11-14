import pytest

import agent_platform.integrations.tracing as tracing_module


class TestTracingInitExports:
    def test_langfuse_tracer_is_lazily_importable(self):
        cls = tracing_module.LangfuseTracer
        assert cls.__name__ == "LangfuseTracer"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            tracing_module.NotATracer

    def test_dir_includes_providers(self):
        assert "LangfuseTracer" in dir(tracing_module)

    def test_alias_resolves_to_the_langfuse_tracer(self):
        assert tracing_module.PROVIDER_ALIASES == {"langfuse": "LangfuseTracer"}
