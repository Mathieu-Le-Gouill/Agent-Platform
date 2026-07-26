from typing import Any


def assert_default_construction(provider_cls: type, config_cls: type) -> None:
    """Assert a provider constructed with no args has a config of the expected type."""
    provider = provider_cls()
    assert isinstance(provider._default_config(), config_cls)


def assert_custom_construction_stored(provider_cls: type, credentials: Any) -> None:
    """Assert a provider constructed with explicit credentials stores it
    as-is (identity, not copies)."""
    provider = provider_cls(credentials=credentials)
    assert provider._credentials is credentials
