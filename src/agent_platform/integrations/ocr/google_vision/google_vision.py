from typing import Sequence
from uuid import UUID, uuid4
import asyncio

from google.cloud import vision

from agent_platform.integrations.credentials import (
    GoogleVisionCredentials,
)
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.errors import ProviderError, error_logged, with_retry


def _from_google_vision(
    response, document_id: UUID, min_confidence: float
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
                        metadata={
                            "confidence": avg_conf,
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

    def _default_config(self) -> GoogleVisionConfig:
        return GoogleVisionConfig()

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

        if self._credentials.credentials_path:
            client = vision.ImageAnnotatorClient.from_service_account_file(
                self._credentials.credentials_path,
            )
        else:
            client = vision.ImageAnnotatorClient()

        document_bytes = await asyncio.to_thread(self._load_bytes, source)
        image = vision.Image(content=document_bytes)

        response = await asyncio.to_thread(client.document_text_detection, image=image)

        return _from_google_vision(
            response, document_id=doc_id, min_confidence=config.min_confidence
        )

    @staticmethod
    def _load_bytes(source: str) -> bytes:
        if source.startswith("http://") or source.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(source) as resp:
                return resp.read()
        with open(source, "rb") as f:
            return f.read()
