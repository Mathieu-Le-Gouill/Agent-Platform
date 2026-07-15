import pytest

from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat


@pytest.fixture
def simple_text_document() -> TextDocument:
    return TextDocument(
        text="Hello world. This is a test document for chunking.",
        source="test.txt",
        format=DocumentFormat.TXT,
    )


@pytest.fixture
def long_text_document() -> TextDocument:
    return TextDocument(
        text=("The quick brown fox jumps over the lazy dog. " * 50),
        source="long.txt",
        format=DocumentFormat.TXT,
    )
