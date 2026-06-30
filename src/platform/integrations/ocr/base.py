from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence
from models.protocols.text_unit import TextUnit

from integrations.ocr.config import OCRConfig

T_co = TypeVar('T_co', bound=TextUnit, covariant=True)

class BaseOCR(ABC, Generic[T_co]):

    @abstractmethod
    async def extract(
        self,
        source: str,
        config: OCRConfig | None = None,
    ) -> Sequence[T_co]:
        ...