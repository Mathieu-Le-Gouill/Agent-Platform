from pydantic import BaseModel, ConfigDict


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    model: str = ""
    batch_size: int = 32
    dimensions: int | None = None
    timeout: float | None = None
