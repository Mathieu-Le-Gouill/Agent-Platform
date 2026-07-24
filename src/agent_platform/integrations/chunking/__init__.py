from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "RecursiveChunkerProvider": "agent_platform.integrations.chunking.recursive.recursive",
    "MarkdownStructureChunkerProvider": "agent_platform.integrations.chunking.markdown.markdown",
    "HTMLStructureChunkerProvider": "agent_platform.integrations.chunking.html.html",
    "LatexChunkerProvider": "agent_platform.integrations.chunking.latex.latex",
    "PDFStructureChunkerProvider": "agent_platform.integrations.chunking.pdf.pdf",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
