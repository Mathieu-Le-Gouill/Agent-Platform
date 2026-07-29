from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar, cast

from agent_platform.core.errors import ProviderError, error_logged
from agent_platform.core.interfaces.moderation.base import BaseModerationProvider
from agent_platform.core.interfaces.moderation.config import ModerationConfig
from agent_platform.core.interfaces.moderation.response import ModerationResult
from agent_platform.core.retry import with_retry

ConfigT = TypeVar("ConfigT", bound=ModerationConfig)
# The async client (e.g. `AsyncOpenAI`), used by `amoderate`/`_invoke_async`.
AsyncClientT = TypeVar("AsyncClientT")
# The sync client, used by `moderate`/`_invoke_sync`. Equal to `AsyncClientT`
# for vendors with a single client class serving both.
SyncClientT = TypeVar("SyncClientT")
# The vendor's raw moderation result, consumed by `_from_native`.
ResultT = TypeVar("ResultT")


class NativeModerationProvider(
    BaseModerationProvider[ConfigT],
    Generic[ConfigT, AsyncClientT, SyncClientT, ResultT],
):
    """Shared request/response plumbing for a network-bound moderation vendor.

    Mirrors `integrations/reranking/_base.py::NativeReranker`'s dual-client
    shape (OpenAI ships both `OpenAI`/`AsyncOpenAI`), minus the batch/index
    mapping reranking needs, since moderation is one-text-in, one-result-out.
    """

    @abstractmethod
    def _default_config(self) -> ConfigT: ...

    @abstractmethod
    def _async_client(self, config: ConfigT) -> AsyncClientT: ...

    def _sync_client(self, config: ConfigT) -> SyncClientT:
        return cast(SyncClientT, self._async_client(config))

    @abstractmethod
    def _invoke_sync(self, client: SyncClientT, text: str, config: ConfigT) -> ResultT:
        """Assemble the native request and make the blocking call."""

    @abstractmethod
    async def _invoke_async(
        self, client: AsyncClientT, text: str, config: ConfigT
    ) -> ResultT:
        """Assemble the native request and make the async call."""

    @abstractmethod
    def _from_native(self, result: ResultT) -> ModerationResult: ...

    def moderate(self, text: str, config: ConfigT | None = None) -> ModerationResult:
        config = config or self._default_config()
        client = self._sync_client(config)
        result = self._invoke_sync(client, text, config)
        return self._from_native(result)

    @error_logged(re_raise=ProviderError, message="Moderation request failed")
    @with_retry()
    async def amoderate(
        self, text: str, config: ConfigT | None = None
    ) -> ModerationResult:
        config = config or self._default_config()
        client = self._async_client(config)
        result = await self._invoke_async(client, text, config)
        return self._from_native(result)
