from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.translation.base import BaseTranslator, ConfigT
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language

# The vendor's client object, however it's constructed (e.g. `deepl.Translator`,
# `google_translate.Client`, `TextTranslationClient`).
ClientT = TypeVar("ClientT")


class NativeTranslator(BaseTranslator[ConfigT], Generic[ConfigT, ClientT]):
    """Shared sync/async plumbing for translation providers. None of the
    vendor SDKs in this domain (deepl, google-cloud-translate,
    azure-ai-translation-text) ship a native async client, so `atranslate`
    is templated once here as `asyncio.to_thread(self.translate, ...)` with
    retry/error-translation applied; each subclass only owns client
    construction and the blocking request/response mapping via `_invoke`.
    """

    @abstractmethod
    def _default_config(self) -> ConfigT: ...

    @abstractmethod
    def _client(self, config: ConfigT) -> ClientT: ...

    @abstractmethod
    def _invoke(
        self,
        client: ClientT,
        content: TextChunk,
        target: Language,
        source: Language | None,
        config: ConfigT,
    ) -> TextChunk:
        """Assemble the native request and make the blocking call, returning
        the mapped `TextChunk`."""

    def translate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
        config: ConfigT | None = None,
    ) -> TextChunk:
        config = config or self._default_config()
        client = self._client(config)
        return self._invoke(client, content, target, source, config)

    @error_logged(re_raise=ProviderError, message="Translation failed")
    @with_retry()
    async def atranslate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
        config: ConfigT | None = None,
    ) -> TextChunk:
        return await asyncio.to_thread(self.translate, content, target, source, config)
