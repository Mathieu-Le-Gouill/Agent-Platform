from typing import Sequence
from urllib.request import urlopen
from uuid import UUID, uuid4
import asyncio

import boto3

from agent_platform.integrations.credentials import AWSTextractCredentials
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.errors import ProviderError, error_logged, with_retry


def _from_textract(
    response: dict, document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for block in response.get("Blocks", []):
        if block.get("BlockType") != "LINE":
            continue

        conf = block.get("Confidence", 0) or 0
        if conf < min_confidence:
            continue

        text = block.get("Text", "") or ""
        bbox = block.get("Geometry", {}).get("BoundingBox", {})

        chunks.append(
            TextChunk(
                id=uuid4(),
                document_id=document_id,
                text=text.strip(),
                metadata={
                    "confidence": conf,
                    "bbox": {
                        "left": bbox.get("Left"),
                        "top": bbox.get("Top"),
                        "width": bbox.get("Width"),
                        "height": bbox.get("Height"),
                    },
                    "page": block.get("Page"),
                },
            )
        )

    return chunks


class AWSTextractOCR(BaseOCR[AWSTextractConfig]):
    def __init__(self, credentials: AWSTextractCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else AWSTextractCredentials()
        )

    def _default_config(self) -> AWSTextractConfig:
        return AWSTextractConfig()

    @error_logged(re_raise=ProviderError, message="OCR extraction failed")
    @with_retry()
    async def extract(
        self,
        source: str,
        config: AWSTextractConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        doc_id = document_id or uuid4()
        document_bytes = await asyncio.to_thread(self._load_bytes, source)

        client = boto3.client(
            "textract",
            region_name=config.region_name,
            aws_access_key_id=self._credentials.aws_access_key_id,
            aws_secret_access_key=self._credentials.aws_secret_access_key,
        )

        response = await asyncio.to_thread(
            client.detect_document_text,
            Document={"Bytes": document_bytes},
        )

        return _from_textract(
            response, document_id=doc_id, min_confidence=config.min_confidence
        )

    @staticmethod
    def _load_bytes(source: str) -> bytes:
        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source) as response:
                return response.read()
        with open(source, "rb") as f:
            return f.read()
