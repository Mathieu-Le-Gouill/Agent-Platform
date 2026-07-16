import pytest

pytest.importorskip("weaviate")
pytest.importorskip("langchain_weaviate")

from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig
from agent_platform.integrations.vector_store.weaviate.weaviate import WeaviateStore


@pytest.fixture
def provider():
    return WeaviateStore.__new__(WeaviateStore)


class TestWeaviateSearchKwargs:
    def test_no_filter_no_namespace_returns_empty(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(), None)
        assert kwargs == {}

    def test_filter_uses_filters_kwarg_not_filter(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(), {"source": "doc.txt"})
        assert "filters" in kwargs
        assert "filter" not in kwargs

    def test_single_condition_filter(self, provider):
        from weaviate.collections.classes.filters import _FilterValue

        kwargs = provider._search_kwargs(WeaviateConfig(), {"source": "doc.txt"})
        assert isinstance(kwargs["filters"], _FilterValue)

    def test_multi_key_filter_combines_with_all_of(self, provider):
        from weaviate.collections.classes.filters import _Filters

        kwargs = provider._search_kwargs(
            WeaviateConfig(), {"source": "doc.txt", "language": "en"}
        )
        assert isinstance(kwargs["filters"], _Filters)

    def test_namespace_maps_to_tenant_kwarg(self, provider):
        kwargs = provider._search_kwargs(
            WeaviateConfig(namespace="tenant-a"), None
        )
        assert kwargs == {"tenant": "tenant-a"}

    def test_no_namespace_omits_tenant(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(), None)
        assert "tenant" not in kwargs

    def test_build_client_enables_multi_tenancy_when_namespace_set(
        self, provider, monkeypatch
    ):
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        captured = {}

        class FakeWeaviateVectorStore:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "WeaviateVectorStore", FakeWeaviateVectorStore)
        provider._credentials = mod.WeaviateCredentials(api_key=None)

        class FakeClient:
            pass

        monkeypatch.setattr(mod.weaviate, "connect_to_local", lambda url: FakeClient())

        provider._embeddings = None
        provider._build_client(WeaviateConfig(namespace="tenant-a"))

        assert captured["use_multi_tenancy"] is True

    def test_build_client_disables_multi_tenancy_without_namespace(
        self, provider, monkeypatch
    ):
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        captured = {}

        class FakeWeaviateVectorStore:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "WeaviateVectorStore", FakeWeaviateVectorStore)
        provider._credentials = mod.WeaviateCredentials(api_key=None)

        class FakeClient:
            pass

        monkeypatch.setattr(mod.weaviate, "connect_to_local", lambda url: FakeClient())

        provider._embeddings = None
        provider._build_client(WeaviateConfig())

        assert captured["use_multi_tenancy"] is False
