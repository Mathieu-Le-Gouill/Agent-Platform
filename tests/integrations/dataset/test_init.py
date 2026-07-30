import pytest

pytest.importorskip("datasets")

from agent_platform.integrations.dataset import _PROVIDERS, PROVIDER_ALIASES
from agent_platform.integrations.dataset.huggingface.provider import (
    HuggingFaceDatasetProvider,
)


class TestDatasetInitExports:
    def test_huggingface_provider_is_lazily_importable(self):
        import agent_platform.integrations.dataset as dataset_module

        assert dataset_module.HuggingFaceDatasetProvider is HuggingFaceDatasetProvider

    def test_providers_map(self):
        assert _PROVIDERS["HuggingFaceDatasetProvider"] == (
            "agent_platform.integrations.dataset.huggingface.provider"
        )

    def test_provider_aliases(self):
        assert PROVIDER_ALIASES["huggingface"] == "HuggingFaceDatasetProvider"

    def test_dir_includes_providers(self):
        import agent_platform.integrations.dataset as dataset_module

        assert "HuggingFaceDatasetProvider" in dir(dataset_module)

    def test_unknown_attribute_raises(self):
        import agent_platform.integrations.dataset as dataset_module

        with pytest.raises(AttributeError):
            dataset_module.NotAProvider
