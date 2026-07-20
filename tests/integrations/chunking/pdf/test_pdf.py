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


def test_default_config_overlap_all_defaults_true():
    cfg = PDFChunkerConfig()
    assert cfg.overlap_all is True


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_forwards_overlap_all_by_default(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)

    provider.chunk([doc], None)

    _, kwargs = mock_chunk_by_title.call_args
    assert kwargs["overlap_all"] is True
    assert kwargs["overlap"] == 64


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_forwards_overlap_all_false_when_configured(
    mock_partition, mock_chunk_by_title
):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)
    config = PDFChunkerConfig(overlap_all=False)

    provider.chunk([doc], config)

    _, kwargs = mock_chunk_by_title.call_args
    assert kwargs["overlap_all"] is False


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_forwards_include_orig_elements(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)
    config = PDFChunkerConfig(include_orig_elements=True)

    provider.chunk([doc], config)

    _, kwargs = mock_chunk_by_title.call_args
    assert kwargs["include_orig_elements"] is True


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_forwards_max_tokens_when_set(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)
    config = PDFChunkerConfig(max_tokens=256)

    provider.chunk([doc], config)

    _, kwargs = mock_chunk_by_title.call_args
    assert kwargs["max_tokens"] == 256


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_omits_max_tokens_when_unset(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)

    provider.chunk([doc], None)

    _, kwargs = mock_chunk_by_title.call_args
    assert "max_tokens" not in kwargs


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_forwards_table_options_when_true(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)
    config = PDFChunkerConfig(
        skip_table_chunking=True,
        repeat_table_headers=True,
        isolate_table=True,
    )

    provider.chunk([doc], config)

    _, kwargs = mock_chunk_by_title.call_args
    assert kwargs["skip_table_chunking"] is True
    assert kwargs["repeat_table_headers"] is True
    assert kwargs["isolate_table"] is True


@patch("agent_platform.integrations.chunking.pdf.pdf.chunk_by_title")
@patch("agent_platform.integrations.chunking.pdf.pdf.partition_pdf")
def test_chunk_omits_table_options_when_false(mock_partition, mock_chunk_by_title):
    mock_partition.return_value = ["element"]
    mock_chunk_by_title.return_value = [_FakeSection("Section body.")]

    provider = PDFStructureChunkerProvider()
    doc = TextDocument(text="", source="doc.pdf", format=DocumentFormat.PDF)

    provider.chunk([doc], None)

    _, kwargs = mock_chunk_by_title.call_args
    assert "skip_table_chunking" not in kwargs
    assert "repeat_table_headers" not in kwargs
    assert "isolate_table" not in kwargs
