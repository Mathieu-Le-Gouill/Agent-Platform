from __future__ import annotations
from agent_platform.core.interfaces.loader.config import LoaderConfig


class UnstructuredLoaderConfig(LoaderConfig):
    mode: str = "elements"
    chunking_strategy: str | None = None
