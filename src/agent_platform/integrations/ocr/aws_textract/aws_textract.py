import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import boto3
from botocore.client import BaseClient

from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score, ScoreKind
from agent_platform.integrations.credentials import AWSTextractCredentials
from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.integrations.ocr.utils import load_bytes


def _from_textract(
    response: dict[str, Any], document_id: UUID, min_confidence: float
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
                confidence=Score(
                    value=conf, kind=ScoreKind.CONFIDENCE, low=0, high=100
                ),
                bbox=BoundingBox(
                    x=bbox.get("Left", 0.0),
                    y=bbox.get("Top", 0.0),
                    width=bbox.get("Width", 0.0),
                    height=bbox.get("Height", 0.0),
                    normalized=True,
                ),
                metadata={
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
        self._clients: dict[str, Any] = {}

    def _default_config(self) -> AWSTextractConfig:
        return AWSTextractConfig()

    def _get_client(self, region_name: str) -> BaseClient:
        client = self._clients.get(region_name)
        if client is None:
            client = boto3.client(
                "textract",
                region_name=region_name,
                aws_access_key_id=self._credentials.aws_access_key_id,
                aws_secret_access_key=self._credentials.aws_secret_access_key,
            )
            self._clients[region_name] = client
        return client

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
        document_bytes = await asyncio.to_thread(load_bytes, source)

        client = self._get_client(config.region_name)

        response = await asyncio.to_thread(
            client.detect_document_text,
            Document={"Bytes": document_bytes},
        )

        return _from_textract(
            response, document_id=doc_id, min_confidence=config.min_confidence
        )
