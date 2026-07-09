from pydantic import BaseModel, ConfigDict


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_size: int = 512
    chunk_overlap: int = 64


class RecursiveChunkerConfig(ChunkerConfig):
    separators: list[str] | None = ["\n\n", "\n", " ", ""]
    add_start_index: bool = False
