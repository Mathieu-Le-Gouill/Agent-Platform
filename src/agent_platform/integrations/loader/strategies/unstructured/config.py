from __future__ import annotations

from agent_platform.core.interfaces.loader.config import LoaderConfig


class UnstructuredLoaderConfig(LoaderConfig):
    # Partitioning output mode ("elements", "single", "paged"). https://docs.unstructured.io
    mode: str = "elements"
    # Chunking strategy applied after partitioning ("basic", "by_title", ...); None disables chunking. https://docs.unstructured.io
    chunking_strategy: str | None = None
    # Partitioning strategy ("auto", "fast", "hi_res", "ocr_only"); None lets the
    # library pick its own default. https://docs.unstructured.io
    strategy: str | None = None
