from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Sequence
from uuid import UUID, uuid4

from mistralai import Mistral

from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.integrations.ocr.mistral.config import MistralOCRConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.errors import (
    MissingCredentialError,
    ProviderError,
    error_logged,
    with_retry,
)


_MIME_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}


class MistralOCR(BaseOCR[MistralOCRConfig]):
    def __init__(self, credentials: MistralCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else MistralCredentials()
        )

    def _default_config(self) -> MistralOCRConfig:
        return MistralOCRConfig()

    @error_logged(re_raise=ProviderError, message="OCR extraction failed")
    @with_retry()
    async def extract(
        self,
        source: str,
        config: MistralOCRConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if not self._credentials.api_key:
            raise MissingCredentialError("Mistral API key is required")

        doc_id = document_id or uuid4()
        client = Mistral(api_key=self._credentials.api_key.get_secret_value())

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

        try:
            response = await asyncio.to_thread(
                client.ocr.process,
                model=config.model,
                document=document,
                include_image_base64=False,
            )
        except Exception as exc:
            raise ProviderError(f"Mistral OCR failed: {exc}") from exc

        return _from_mistral(response, doc_id)


def _from_mistral(response, document_id: UUID) -> list[TextChunk]:
    chunks = []
    for page in response.pages:
        text = (page.markdown or "").strip()
        if text:
            chunks.append(
                TextChunk(
                    id=uuid4(),
                    document_id=document_id,
                    text=text,
                    index=page.index,
                    metadata={"page": page.index},
                )
            )
    return chunks
