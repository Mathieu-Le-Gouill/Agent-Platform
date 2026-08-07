import pytest

import agent_platform.integrations.tracing as tracing_module


class TestTracingInitExports:
    def test_otel_tracer_is_lazily_importable(self):
        cls = tracing_module.OTelTracer
        assert cls.__name__ == "OTelTracer"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            tracing_module.NotATracer

    def test_dir_includes_providers(self):
        assert "OTelTracer" in dir(tracing_module)

    def test_aliases_resolve_to_the_otel_tracer(self):
        assert tracing_module.PROVIDER_ALIASES["otel"] == "OTelTracer"
        assert tracing_module.PROVIDER_ALIASES["opentelemetry"] == "OTelTracer"
