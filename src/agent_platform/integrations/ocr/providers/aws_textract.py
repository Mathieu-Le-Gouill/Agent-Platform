from typing import Sequence
from urllib.request import urlopen
from uuid import UUID, uuid4
import asyncio

import boto3

from agent_platform.integrations.ocr.base import BaseOCR
from agent_platform.integrations.ocr.config import OCRConfig, AWSTextractConfig
from agent_platform.models.chunk import Chunk

from agent_platform.bridges.aws.chunk import from_textract


class AWSTextractOCR(BaseOCR):

    def __init__(
        self,
        config: AWSTextractConfig | None = None,
    ) -> None:

        self._config = config or AWSTextractConfig()
        self._client = self._build_client()


    def _build_client(self):

        return boto3.client(
            "textract",
            region_name=self._config.region_name,
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
        )


    async def extract(
        self,
        source: str,
        config: OCRConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[Chunk]:

        cfg = config or self._config
        doc_id = document_id or uuid4()

        document_bytes = await asyncio.to_thread(self._load_bytes, source)

        response = await asyncio.to_thread(
            self._client.detect_document_text,
            Document={"Bytes": document_bytes},
        )

        return from_textract(
            response,
            document_id=doc_id,
            min_confidence=cfg.min_confidence,
        )


    @staticmethod
    def _load_bytes(source: str) -> bytes:

        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source) as response:
                return response.read()

        with open(source, "rb") as f:
            return f.read()