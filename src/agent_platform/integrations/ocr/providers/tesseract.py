from pathlib import Path
from typing import Sequence
from urllib.request import urlopen
from uuid import UUID, uuid4
import asyncio

import pytesseract
from PIL import Image

from agent_platform.integrations.ocr.base import BaseOCR
from agent_platform.integrations.ocr.config import TesseractConfig
from agent_platform.models.chunk import TextChunk


def _from_tesseract(
    data: dict, document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    n = len(data.get("text", []))

    for i in range(n):
        text = (data.get("text") or [])[i] or ""
        conf = float((data.get("conf") or [0])[i] or 0)

        if not text.strip():
            continue
        if conf < min_confidence:
            continue

        chunks.append(
            TextChunk(
                id=uuid4(),
                document_id=document_id,
                text=text.strip(),
                metadata={
                    "confidence": conf,
                    "bbox": {
                        "x": int((data.get("left") or [0])[i]),
                        "y": int((data.get("top") or [0])[i]),
                        "w": int((data.get("width") or [0])[i]),
                        "h": int((data.get("height") or [0])[i]),
                    },
                    "page": (data.get("page_num") or [None] * n)[i],
                    "block_num": (data.get("block_num") or [None] * n)[i],
                },
            )
        )

    return chunks


class TesseractOCR(BaseOCR[TesseractConfig]):
    def __init__(self, config: TesseractConfig | None = None) -> None:
        self._config = config or TesseractConfig()
        if self._config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_cmd

    async def extract(
        self,
        source: str,
        config: TesseractConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        cfg = config or self._config
        doc_id = document_id or uuid4()
        image = await asyncio.to_thread(self._load_image, source)

        data = await asyncio.to_thread(
            pytesseract.image_to_data,
            image,
            lang=cfg.language,
            config=f"--psm {cfg.psm} --oem {cfg.oem}",
            output_type=pytesseract.Output.DICT,
        )

        return _from_tesseract(
            data, document_id=doc_id, min_confidence=cfg.min_confidence
        )

    @staticmethod
    def _load_image(source: str) -> Image.Image:
        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source) as response:
                return Image.open(response).convert("RGB")
        return Image.open(Path(source)).convert("RGB")
