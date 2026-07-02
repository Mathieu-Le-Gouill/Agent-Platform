from pathlib import Path
from typing import Sequence
from urllib.request import urlopen
from uuid import UUID, uuid4
import asyncio

import pytesseract
from PIL import Image

from agent_platform.integrations.ocr.base import BaseOCR
from agent_platform.integrations.ocr.config import OCRConfig, TesseractConfig
from agent_platform.models.chunk import Chunk

from agent_platform.bridges.tesseract.chunk import from_tesseract


class TesseractOCR(BaseOCR):

    def __init__(
        self,
        config: TesseractConfig | None = None,
    ) -> None:

        self._config = config or TesseractConfig()

        if self._config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_cmd


    async def extract(
        self,
        source: str,
        config: OCRConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[Chunk]:

        cfg = config or self._config
        doc_id = document_id or uuid4()

        image = await asyncio.to_thread(self._load_image, source)

        data = await asyncio.to_thread(
            pytesseract.image_to_data,
            image,
            lang=cfg.language,
            config=self._tesseract_config(cfg),
            output_type=pytesseract.Output.DICT,
        )

        return from_tesseract(
            data,
            document_id=doc_id,
            min_confidence=cfg.min_confidence,
        )


    @staticmethod
    def _tesseract_config(cfg: OCRConfig) -> str:

        psm = getattr(cfg, "psm", 3)
        oem = getattr(cfg, "oem", 3)

        return f"--psm {psm} --oem {oem}"


    @staticmethod
    def _load_image(source: str) -> Image.Image:

        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source) as response:
                return Image.open(response).convert("RGB")

        return Image.open(Path(source)).convert("RGB")