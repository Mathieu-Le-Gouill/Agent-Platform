from pydantic import Field, SecretStr

from agent_platform.core.credentials import Credentials
from agent_platform.utils.env import from_env, secret_from_env

__all__ = [
    "AnthropicCredentials",
    "AWSTextractCredentials",
    "AzureTranslatorCredentials",
    "ChromaCredentials",
    "CohereCredentials",
    "DeepgramCredentials",
    "DeepLCredentials",
    "GoogleCredentials",
    "GoogleTranslateCredentials",
    "GoogleVisionCredentials",
    "HuggingFaceCredentials",
    "JinaCredentials",
    "MidjourneyCredentials",
    "MistralCredentials",
    "OllamaCredentials",
    "OpenAICredentials",
    "PicoVoiceCredentials",
    "PineconeCredentials",
    "QdrantCredentials",
    "VoyageCredentials",
    "WeaviateCredentials",
]


class AnthropicCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("ANTHROPIC_API_KEY")
    )


class AWSTextractCredentials(Credentials, frozen=True):
    aws_access_key_id: str | None = Field(
        default_factory=lambda: from_env("AWS_ACCESS_KEY_ID")
    )
    aws_secret_access_key: str | None = Field(
        default_factory=lambda: from_env("AWS_SECRET_ACCESS_KEY")
    )


class AzureTranslatorCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("AZURE_TRANSLATOR_KEY")
    )
    region: str | None = Field(
        default_factory=lambda: from_env("AZURE_TRANSLATOR_REGION")
    )


class ChromaCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("CHROMA_API_KEY")
    )


class CohereCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("COHERE_API_KEY")
    )


class DeepgramCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("DEEPGRAM_API_KEY")
    )


class DeepLCredentials(Credentials, frozen=True):
    auth_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("DEEPL_AUTH_KEY")
    )


class GoogleCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env(["GOOGLE_API_KEY", "GEMINI_API_KEY"])
    )


class GoogleTranslateCredentials(Credentials, frozen=True):
    credentials_path: str | None = Field(
        default_factory=lambda: from_env("GOOGLE_APPLICATION_CREDENTIALS")
    )


class GoogleVisionCredentials(Credentials, frozen=True):
    credentials_path: str | None = Field(
        default_factory=lambda: from_env("GOOGLE_APPLICATION_CREDENTIALS")
    )


class HuggingFaceCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("HF_TOKEN")
    )


class JinaCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("JINA_API_KEY")
    )


class MidjourneyCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("MIDJOURNEY_API_KEY")
    )


class MistralCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("MISTRAL_API_KEY")
    )


class OllamaCredentials(Credentials, frozen=True):
    pass


class OpenAICredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("OPENAI_API_KEY")
    )
    organization: str | None = Field(
        default_factory=lambda: from_env(["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"])
    )


class PicoVoiceCredentials(Credentials, frozen=True):
    access_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("PICOVOICE_ACCESS_KEY")
    )


class PineconeCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("PINECONE_API_KEY")
    )


class QdrantCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("QDRANT_API_KEY")
    )


class VoyageCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("VOYAGE_API_KEY")
    )


class WeaviateCredentials(Credentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("WEAVIATE_API_KEY")
    )
