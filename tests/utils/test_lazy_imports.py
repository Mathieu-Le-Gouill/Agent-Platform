import pytest

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors


class TestMakeLazyProviderAccessors:
    def make_module_globals(self) -> dict:
        return {"__name__": "fake.module", "__doc__": None}

    def test_getattr_resolves_registered_provider(self):
        providers = {"filter_by_score": "agent_platform.utils.score"}
        __getattr__, __dir__ = make_lazy_provider_accessors(
            providers, self.make_module_globals()
        )

        from agent_platform.utils.score import filter_by_score

        assert __getattr__("filter_by_score") is filter_by_score

    def test_getattr_raises_attribute_error_for_unknown_name(self):
        __getattr__, __dir__ = make_lazy_provider_accessors(
            {}, self.make_module_globals()
        )

        with pytest.raises(AttributeError, match="fake.module.*Unknown"):
            __getattr__("Unknown")

    def test_dir_includes_providers_and_module_globals(self):
        providers = {"SomeProvider": "some.module.path"}
        module_globals = self.make_module_globals()
        __getattr__, __dir__ = make_lazy_provider_accessors(providers, module_globals)

        result = __dir__()

        assert result == sorted(list(module_globals.keys()) + list(providers.keys()))

    def test_getattr_does_not_import_unregistered_providers_eagerly(self):
        providers = {"Nope": "agent_platform.does.not.exist"}
        __getattr__, __dir__ = make_lazy_provider_accessors(
            providers, self.make_module_globals()
        )

        with pytest.raises(ModuleNotFoundError):
            __getattr__("Nope")
