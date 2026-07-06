from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ChunkerConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(slots=True, frozen=True)
class RecursiveChunkerConfig(ChunkerConfig):
    separators: tuple[str, ...] = (
        "\n\n",
        "\n",
        " ",
        "",
    )
