import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from google.cloud import vision

from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import (
    GoogleVisionCredentials,
)
from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig
from agent_platform.integrations.ocr.utils import load_bytes


def _from_google_vision(
    response: vision.AnnotateImageResponse, document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    pages = response.full_text_annotation.pages

    for page_num, page in enumerate(pages):
        for block in page.blocks:
            for paragraph in block.paragraphs:
                words = paragraph.words
                if not words:
                    continue

                text = " ".join(
                    "".join(symbol.text for symbol in word.symbols) for word in words
                )
                confs = [
                    symbol.confidence
                    for word in words
                    for symbol in word.symbols
                    if symbol.confidence
                ]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                if avg_conf < min_confidence:
                    continue

                chunks.append(
                    TextChunk(
                        id=uuid4(),
                        document_id=document_id,
                        text=text.strip(),
                        confidence=Score.confidence(avg_conf),
                        metadata={
                            "page": page_num,
                        },
                    )
                )

    return chunks


class GoogleVisionOCR(BaseOCR[GoogleVisionConfig]):
    def __init__(self, credentials: GoogleVisionCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else GoogleVisionCredentials()
        )
        self._client: vision.ImageAnnotatorClient | None = None

    def _default_config(self) -> GoogleVisionConfig:
        return GoogleVisionConfig()

    def _get_client(self) -> vision.ImageAnnotatorClient:
        if self._client is None:
            if self._credentials.credentials_path:
                self._client = vision.ImageAnnotatorClient.from_service_account_file(
                    self._credentials.credentials_path,
                )
            else:
                self._client = vision.ImageAnnotatorClient()
        return self._client

    @error_logged(re_raise=ProviderError, message="OCR extraction failed")
    @with_retry()
    async def extract(
        self,
        source: str,
        config: GoogleVisionConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        doc_id = document_id or uuid4()

        client = self._get_client()

        document_bytes = await asyncio.to_thread(load_bytes, source)
        image = vision.Image(content=document_bytes)

        kwargs: dict[str, Any] = {"image": image}
        if config.language_hints:
            kwargs["image_context"] = {"language_hints": config.language_hints}

        method = (
            client.text_detection
            if config.feature_type == "TEXT_DETECTION"
            else client.document_text_detection
        )
        response = await asyncio.to_thread(method, **kwargs)

        return _from_google_vision(
            response, document_id=doc_id, min_confidence=config.min_confidence
        )
