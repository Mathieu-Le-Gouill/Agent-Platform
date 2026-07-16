import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("langchain_qdrant")

from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig
from agent_platform.integrations.vector_store.qdrant.qdrant import (
    QdrantVectorStoreProvider,
)


@pytest.fixture
def provider():
    return QdrantVectorStoreProvider.__new__(QdrantVectorStoreProvider)


class TestQdrantSearchKwargs:
    def test_no_filter_returns_none(self, provider):
        kwargs = provider._search_kwargs(QdrantConfig(), None)
        assert kwargs == {"filter": None}

    def test_empty_filter_returns_none(self, provider):
        kwargs = provider._search_kwargs(QdrantConfig(), {})
        assert kwargs == {"filter": None}

    def test_filter_builds_qdrant_filter_model(self, provider):
        from qdrant_client import models

        kwargs = provider._search_kwargs(QdrantConfig(), {"source": "doc.txt"})
        qfilter = kwargs["filter"]
        assert isinstance(qfilter, models.Filter)
        assert len(qfilter.must) == 1
        condition = qfilter.must[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "source"
        assert condition.match == models.MatchValue(value="doc.txt")

    def test_multi_key_filter_produces_multiple_conditions(self, provider):
        kwargs = provider._search_kwargs(
            QdrantConfig(), {"source": "doc.txt", "language": "en"}
        )
        assert len(kwargs["filter"].must) == 2
