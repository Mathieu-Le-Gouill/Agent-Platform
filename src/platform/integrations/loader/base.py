from abc import ABC, abstractmethod
from typing import AsyncGenerator
import asyncio

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document as LCDocument

from bridges.langchain.document import from_langchain
from models.document import Document


class BaseLoader(ABC):

    @abstractmethod
    def _loader(self, source: str, **kwargs) -> UnstructuredFileLoader:
        ...


    async def load(self, source: str,  **kwargs) -> Document:

        docs: list[LCDocument] = await asyncio.to_thread(
            self._loader(source, **kwargs).load
        )
        if not docs:
            return Document(source=source)

        return from_langchain(docs, source)


    async def load_many(self, sources: list[str], **kwargs) -> AsyncGenerator[Document, None]:

        results: list[Document | BaseException] = await asyncio.gather(
            *[self.load(s, **kwargs) for s in sources],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                continue
            yield result

