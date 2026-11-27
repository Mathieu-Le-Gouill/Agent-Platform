import pytest

from agent_platform.core.interfaces.reranking.base import BaseReranker


class _MinimalReranker(BaseReranker):
    def rerank(self, query, items, config=None):
        return list(items)

    async def arerank(self, query, items, config=None):
        return list(items)


def test_default_config_is_optional_for_subclasses():
    with pytest.raises(NotImplementedError):
        _MinimalReranker().default_config()
