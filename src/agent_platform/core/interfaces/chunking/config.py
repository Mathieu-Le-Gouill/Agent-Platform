from pydantic import BaseModel, ConfigDict


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    chunk_size: int = 512
    chunk_overlap: int = 64
