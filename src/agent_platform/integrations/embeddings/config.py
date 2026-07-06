from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    batch_size: int = 32
    normalize: bool = True
    truncate: bool = True

    dimensions: int | None = None

    device: str | None = None
    show_progress: bool = False


@dataclass(slots=True, frozen=True)
class OpenAIEmbeddingConfig(EmbeddingConfig):
    organization: str | None = None
    timeout: float | None = None


@dataclass(slots=True, frozen=True)
class SentenceTransformerConfig(EmbeddingConfig):
    model_kwargs: dict | None = None
    encode_kwargs: dict | None = None


@dataclass(slots=True, frozen=True)
class MistralEmbeddingConfig(EmbeddingConfig):
    timeout: int | None = None


@dataclass(slots=True, frozen=True)
class OllamaEmbeddingConfig(EmbeddingConfig):
    base_url: str = "http://localhost:11434"
