from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "RecursiveChunkerProvider": "agent_platform.integrations.chunking.recursive.provider",
    "MarkdownStructureChunkerProvider": "agent_platform.integrations.chunking.markdown.provider",
    "HTMLStructureChunkerProvider": "agent_platform.integrations.chunking.html.provider",
    "LatexChunkerProvider": "agent_platform.integrations.chunking.latex.provider",
    "PDFStructureChunkerProvider": "agent_platform.integrations.chunking.pdf.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
