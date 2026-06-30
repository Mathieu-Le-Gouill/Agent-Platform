from typing import Protocol, Sequence, TypeVar

from models.protocols.text_unit import TextUnit
from integrations.ocr.config import OCRConfig

T_co = TypeVar("T_co", bound=TextUnit, covariant=True)

class OCR(Protocol[T_co]):

    async def extract(
        self,
        source: str,
        config: OCRConfig | None = None,
    ) -> Sequence[T_co]:
        ...