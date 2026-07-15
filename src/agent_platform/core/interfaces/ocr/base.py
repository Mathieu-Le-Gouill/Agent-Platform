from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar
from uuid import UUID

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.ocr.config import OCRConfig
from agent_platform.core.schemas.chunk import TextChunk

OCRConfigT = TypeVar("OCRConfigT", bound=OCRConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseOCR(ABC, Generic[CredentialsT, OCRConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    async def extract(
        self,
        source: str,
        config: OCRConfigT | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]: ...
