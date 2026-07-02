from dataclasses import dataclass
from pydantic import SecretStr
from os import getenv


@dataclass(slots=True, frozen=True)
class RerankerConfig:

    top_k: int | None = None

    return_scores: bool = False

    batch_size: int = 32

    normalize_scores: bool = True


@dataclass(slots=True, frozen=True)
class CohereRerankerConfig(RerankerConfig):
    api_key: SecretStr | None = (
        SecretStr(v) if (v := getenv("COHERE_API_KEY")) else None
    )


@dataclass(slots=True, frozen=True)
class JinaRerankerConfig(RerankerConfig):
    api_key: SecretStr | None = (
        SecretStr(v) if (v := getenv("JINA_API_KEY")) else None
    )


@dataclass(slots=True, frozen=True)
class HuggingFaceRerankerConfig(RerankerConfig):
    device: str | None = None