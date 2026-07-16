from agent_platform.core.schemas.config import ProviderConfig


class EmbeddingConfig(ProviderConfig):
    model: str = ""
    batch_size: int = 32
    dimensions: int | None = None
    timeout: float | None = None
