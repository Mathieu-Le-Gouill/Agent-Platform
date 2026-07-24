from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "RecursiveChunkerProvider": "agent_platform.integrations.chunking.recursive.recursive",
    "MarkdownStructureChunkerProvider": "agent_platform.integrations.chunking.markdown.markdown",
    "HTMLStructureChunkerProvider": "agent_platform.integrations.chunking.html.html",
    "LatexChunkerProvider": "agent_platform.integrations.chunking.latex.latex",
    "PDFStructureChunkerProvider": "agent_platform.integrations.chunking.pdf.pdf",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
