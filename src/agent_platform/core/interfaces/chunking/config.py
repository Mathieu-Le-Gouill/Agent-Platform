from agent_platform.core.config import ProviderConfig


class ChunkerConfig(ProviderConfig):
    # Target size of each chunk; unit and enforcement are provider-specific
    # (e.g. characters for text splitters, max_characters for element-based
    # chunkers) — each provider reinterprets this against its own splitter library.
    chunk_size: int = 512
    # Amount of overlap between consecutive chunks; whether and how it is
    # applied depends on the provider's underlying splitter library.
    chunk_overlap: int = 64
