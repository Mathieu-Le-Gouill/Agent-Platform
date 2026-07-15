import pytest

from agent_platform.core.registry import ProviderRegistry
from agent_platform.core.errors import NotFoundError


class TestProviderRegistry:
    def test_register_and_get(self):
        registry: ProviderRegistry[int] = ProviderRegistry()
        registry.register("square", int)
        assert registry.get("square") is int

    def test_get_unknown_key_raises(self):
        registry: ProviderRegistry = ProviderRegistry()
        with pytest.raises(NotFoundError, match="Unknown provider: missing"):
            registry.get("missing")

    def test_all_empty_on_init(self):
        registry: ProviderRegistry = ProviderRegistry()
        assert registry.all() == {}

    def test_all_returns_copy(self):
        registry: ProviderRegistry = ProviderRegistry()
        registry.register("a", int)
        result = registry.all()
        result["b"] = str
        assert "b" not in registry.all()

    def test_all_returns_all_registered(self):
        registry: ProviderRegistry = ProviderRegistry()
        registry.register("int", int)
        registry.register("str", str)
        registry.register("float", float)
        all_providers = registry.all()
        assert all_providers == {"int": int, "str": str, "float": float}

    def test_register_overwrites_existing(self):
        registry: ProviderRegistry = ProviderRegistry()
        registry.register("key", int)
        registry.register("key", str)
        assert registry.get("key") is str

    def test_multiple_registries_isolated(self):
        r1: ProviderRegistry = ProviderRegistry()
        r2: ProviderRegistry = ProviderRegistry()
        r1.register("x", int)
        with pytest.raises(NotFoundError):
            r2.get("x")

    def test_generic_type_preserved(self):
        registry: ProviderRegistry[str] = ProviderRegistry()
        registry.register("a", str)
        assert registry.get("a") is str
