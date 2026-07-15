from typing import Generic, TypeVar

from agent_platform.core.errors import NotFoundError

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    _providers: dict[str, type[T]]

    def __init__(self) -> None:
        self._providers = {}

    def register(self, name: str, provider: type[T]) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> type[T]:
        provider = self._providers.get(name)
        if provider is None:
            raise NotFoundError(f"Unknown provider: {name}")
        return provider

    def all(self) -> dict[str, type[T]]:
        return dict(self._providers)
