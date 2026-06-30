from abc import abstractmethod
from typing import AsyncIterator, Sequence
import asyncio

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document as LCDocument

from integrations.loader.text_base import BaseTextLoader
from bridges.langchain.document import text_from_langchain
from models.document import TextFile


class UnstructuredBaseLoader(BaseTextLoader):

    @abstractmethod
    def _loader(self, source: str, **kwargs) -> UnstructuredFileLoader: ...


    async def load(self, source: str, **kwargs) -> Sequence[TextFile]:

        docs: list[LCDocument] = await asyncio.to_thread(
            self._loader(source, **kwargs).load
        )
        if not docs:
            return []

        return [text_from_langchain(doc) for doc in docs]


    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[TextFile]]:

        results = await asyncio.gather(
            *[self.load(s, **kwargs) for s in sources],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                continue
            yield result

