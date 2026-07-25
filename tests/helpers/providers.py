from typing import Any
from unittest.mock import AsyncMock, MagicMock


def make_provider_with_mock_client(
    provider_cls: type,
    *,
    async_methods: tuple[str, ...] = (),
    sync_methods: tuple[str, ...] = (),
) -> tuple[Any, MagicMock]:
    """Construct a provider whose `_client(config)` returns a mock with the
    given async/sync method names stubbed. Returns (provider, mock_client)."""
    provider = provider_cls()
    mock_client = MagicMock()
    for name in async_methods:
        setattr(mock_client, name, AsyncMock())
    for name in sync_methods:
        setattr(mock_client, name, MagicMock())
    provider._client = MagicMock(return_value=mock_client)
    return provider, mock_client
