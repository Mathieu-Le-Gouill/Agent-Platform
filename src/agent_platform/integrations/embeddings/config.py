from pydantic import BaseModel, ConfigDict
from enum import Enum



class EmbeddingConfig(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    model: str = ""
    batch_size: int = 32
    dimensions: int | None = None
    timeout: float | None = None


class OpenAIEmbeddingConfig(EmbeddingConfig):
    model: str = "text-embedding-ada-002"
    max_retries: int | None = None
    model_kwargs: dict | None = None


class HuggingFaceEmbeddingMode(str, Enum):
    LOCAL = "local"
    HOSTED = "hosted"

class HuggingFaceEmbeddingConfig(EmbeddingConfig):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    mode: HuggingFaceEmbeddingMode = HuggingFaceEmbeddingMode.LOCAL

    # Local only
    model_kwargs: dict | None = None
    encode_kwargs: dict | None = None

    # Hosted only
    provider: str | None = None


class MistralEmbeddingConfig(EmbeddingConfig):
    model: str = "mistral-embed"
    max_retries: int | None = None
    endpoint: str = "https://api.mistral.ai/v1/"
    wait_time: int | None = None
    max_concurrent_requests: int | None = None


class OllamaEmbeddingConfig(EmbeddingConfig):
    model: str = "nomic-embed-text"
    top_p: float | None = None
    top_k: int | None = None
    temperature: float | None = None