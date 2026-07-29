import asyncio
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen
from uuid import UUID, uuid4

import pytesseract
from PIL import Image

from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.ocr.tesseract.config import TesseractConfig
from agent_platform.integrations.ocr.tesseract.mappers import from_tesseract


class TesseractOCR(BaseOCR[TesseractConfig]):
    def _default_config(self) -> TesseractConfig:
        return TesseractConfig()

    async def extract(
        self,
        source: str,
        config: TesseractConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        doc_id = document_id or uuid4()
        image = await asyncio.to_thread(self._load_image, source)

        data = await asyncio.to_thread(
            pytesseract.image_to_data,
            image,
            lang=config.language,
            config=f"--psm {config.psm} --oem {config.oem}",
            output_type=pytesseract.Output.DICT,
        )

        return from_tesseract(
            data, document_id=doc_id, min_confidence=config.min_confidence
        )

    @staticmethod
    def _load_image(source: str) -> Image.Image:
        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source) as response:
                return Image.open(response).convert("RGB")
        return Image.open(Path(source)).convert("RGB")
