from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from mistralai import Mistral
else:
    try:
        from mistralai import Mistral
    except ImportError:  # pragma: no cover - depends on installed mistralai version
        from mistralai.client import Mistral

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError, error_logged, require_secret
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.ocr.mistral.config import MistralOCRConfig
from agent_platform.integrations.ocr.mistral.mappers import from_mistral

_MIME_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}


class MistralOCR(BaseOCR[MistralOCRConfig]):
    def __init__(
        self,
        credentials: MistralCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, MistralCredentials)
        self._client_options = resolve_client_options(client_options)
        self._client: Mistral | None = None

    def _default_config(self) -> MistralOCRConfig:
        return MistralOCRConfig()

    def _get_client(self, config: MistralOCRConfig) -> Mistral:
        if self._client is None:
            api_key = require_secret(
                self._credentials.api_key, "Mistral API key is required"
            )
            kwargs: dict = {"api_key": api_key.get_secret_value()}
            if self._client_options.base_url:
                kwargs["server_url"] = self._client_options.base_url

            timeout = resolve_timeout(config.timeout, self._client_options)
            if timeout is not None:
                kwargs["timeout_ms"] = int(timeout * 1000)

            # `config.max_retries`/`client_options.max_retries` are intentionally not
            # wired into the native SDK's `retry_config`: unlike OpenAI/Anthropic's
            # simple int attempt count, Mistral's RetryConfig is a time-based backoff
            # (initial_interval/max_interval/exponent/max_elapsed_time) with no
            # attempt-count knob, so there's no faithful translation. The platform's
            # own `@with_retry()` decorator on `extract` already provides equivalent
            # attempt-count-based retry behavior.
            self._client = Mistral(**kwargs)
        return self._client

    @error_logged(re_raise=ProviderError, message="OCR extraction failed")
    @with_retry()
    async def extract(
        self,
        source: str,
        config: MistralOCRConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        client = self._get_client(config)

        doc_id = document_id or uuid4()

        if source.startswith("http://") or source.startswith("https://"):
            document = {"type": "document_url", "document_url": source}
        else:
            data = await asyncio.to_thread(Path(source).read_bytes)
            ext = Path(source).suffix.lstrip(".").lower()
            mime = _MIME_MAP.get(ext, "application/octet-stream")
            b64 = base64.b64encode(data).decode()
            document = {
                "type": "document_url",
                "document_url": f"data:{mime};base64,{b64}",
            }

        kwargs: dict = {
            "model": config.model,
            "document": document,
            "include_image_base64": False,
        }
        if config.confidence_scores_granularity is not None:
            kwargs["confidence_scores_granularity"] = (
                config.confidence_scores_granularity
            )
        if config.pages is not None:
            kwargs["pages"] = config.pages
        if config.table_format is not None:
            kwargs["table_format"] = config.table_format

        try:
            response = await asyncio.to_thread(client.ocr.process, **kwargs)
        except Exception as exc:
            raise ProviderError(f"Mistral OCR failed: {exc}") from exc

        return from_mistral(response, doc_id, min_confidence=config.min_confidence)
