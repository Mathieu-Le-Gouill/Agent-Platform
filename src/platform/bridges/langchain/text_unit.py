# integrations/langchain/converters/text_units.py

from functools import singledispatch

from langchain_core.documents import Document as LCDocument

from bridges.langchain.document import to_langchain as doc_to_lc
from bridges.langchain.chunk import to_langchain as chunk_to_lc

from models.protocols.text_unit import TextUnit
from models.document import Document
from models.chunk import Chunk


@singledispatch
def to_langchain(item: TextUnit) -> list[LCDocument]:
    raise TypeError(
        f"Unsupported type: {type(item)!r}"
    )

@to_langchain.register
def _(item: Document) -> list[LCDocument]:
    return doc_to_lc(item)


@to_langchain.register
def _(item: Chunk) -> list[LCDocument]:
    return [chunk_to_lc(item)]