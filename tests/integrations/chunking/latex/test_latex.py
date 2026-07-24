import pytest

pytest.importorskip("langchain_text_splitters")

from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.langchain_base import LangChainChunker
from agent_platform.integrations.chunking.latex.config import LatexChunkerConfig
from agent_platform.integrations.chunking.latex.latex import LatexChunkerProvider

LATEX_DOC = (
    r"\section{Intro}"
    "\nThis is the introduction paragraph of the document. " * 5 + r"\section{Methods}"
    "\nThis is the methods paragraph describing the approach. " * 5
)


def test_provider_defaults():
    provider = LatexChunkerProvider()
    assert isinstance(provider, LangChainChunker)


def test_default_config_type():
    provider = LatexChunkerProvider()
    cfg = provider._default_config()
    assert isinstance(cfg, LatexChunkerConfig)
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


def test_default_config_new_field_defaults():
    provider = LatexChunkerProvider()
    cfg = provider._default_config()
    assert cfg.keep_separator is True
    assert cfg.is_separator_regex is False
    assert cfg.strip_whitespace is True


def test_splitter_forwards_new_fields_to_from_language(mocker):
    provider = LatexChunkerProvider()
    config = LatexChunkerConfig(keep_separator="end", strip_whitespace=False)

    mock_from_language = mocker.patch(
        "agent_platform.integrations.chunking.latex.latex.RecursiveCharacterTextSplitter.from_language"
    )
    provider._splitter(config)

    _, kwargs = mock_from_language.call_args
    assert kwargs["keep_separator"] == "end"
    assert kwargs["strip_whitespace"] is False
    # is_separator_regex must never be forwarded: from_language() always
    # passes it internally and a duplicate kwarg would raise a TypeError.
    assert "is_separator_regex" not in kwargs


def test_chunk_respects_latex_sections():
    provider = LatexChunkerProvider()
    doc = TextDocument(text=LATEX_DOC, format=DocumentFormat.LATEX)
    config = LatexChunkerConfig(chunk_size=120, chunk_overlap=0)

    chunks = provider.chunk([doc], config)

    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.document_id == doc.id for c in chunks)


def test_chunk_falls_back_to_default_config():
    provider = LatexChunkerProvider()
    doc = TextDocument(text=LATEX_DOC, format=DocumentFormat.LATEX)

    chunks = provider.chunk([doc], None)

    assert len(chunks) >= 1


def test_chunk_accepts_unknown_format():
    provider = LatexChunkerProvider()
    doc = TextDocument(text=LATEX_DOC, format=DocumentFormat.UNKNOWN)

    chunks = provider.chunk([doc], None)

    assert len(chunks) >= 1


@pytest.mark.parametrize(
    "format_",
    [
        DocumentFormat.MARKDOWN,
        DocumentFormat.HTML,
        DocumentFormat.PDF,
        DocumentFormat.TXT,
    ],
)
def test_chunk_rejects_mismatched_format(format_):
    provider = LatexChunkerProvider()
    doc = TextDocument(text=LATEX_DOC, format=format_)

    with pytest.raises(ValidationError):
        provider.chunk([doc], None)
