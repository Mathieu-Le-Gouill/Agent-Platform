from unittest.mock import patch

import pytest

pytest.importorskip("unstructured")

from agent_platform.core.errors import ProviderError, ValidationError
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.pdf.config import PDFChunkerConfig
from agent_platform.integrations.chunking.pdf.pdf import PDFStructureChunkerProvider


class _FakeMetadata:
    def to_dict(self):
        return {"page_number": 1}


class _FakeSection:
    metadata = _FakeMetadata()

    def __init__(self, text: str):
        self._text = text

    def __str__(self) -> str:
        return self._text


def test_provider_defaults():
    provider = PDFStructureChunkerProvider()
    assert isinstance(provider, PDFStructureChunkerProvider)


def test_default_config_type():
    provider = PDFStructureChunkerProvider()
    cfg = provider._default_config()
    assert isinstance(cfg, PDFChunkerConfig)
    assert cfg.multipage_sections is True


def test_chunk_requires_document_source():
    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="", format=DocumentFormat.PDF)

    with pytest.raises(ValidationError):
        provider.chunk([doc], None)


def test_chunk_rejects_mismatched_format():
    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.html", format=DocumentFormat.HTML)

    with pytest.raises(ValidationError):
        provider.chunk([doc], None)


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_accepts_unknown_format(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.UNKNOWN)

    chunks = provider.chunk([doc], None)

    assert len(chunks) == 1


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_maps_sections_to_text_chunks(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [
        _FakeSection("Section one body."),
        _FakeSection("Section two body."),
    ]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)

    chunks = provider.chunk([doc], None)

    assert len(chunks) == 2
    assert chunks[0].text == "Section one body."
    assert chunks[0].format == DocumentFormat.PDF
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].document_id == doc.id
    mock_partition.assert_called_once_with(filename="doc.pdf")


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_wraps_unexpected_errors_as_provider_error(
    mock_partition, mock_chunk_by_title
):
    mock_partition.side_effect = RuntimeError("boom")

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)

    with pytest.raises(ProviderError):
        provider.chunk([doc], None)
