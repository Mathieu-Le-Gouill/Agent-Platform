from agent_platform.core.schemas.config import ProviderConfig


class ChunkerConfig(ProviderConfig):
    chunk_size: int = 512
    chunk_overlap: int = 64
