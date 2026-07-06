from unittest.mock import patch, MagicMock

from agent_platform.integrations.embeddings.config import (
    OpenAIEmbeddingConfig,
    MistralEmbeddingConfig,
    OllamaEmbeddingConfig,
    SentenceTransformerConfig,
)


class TestSentenceTransformerConstructor:
    @patch(
        "agent_platform.integrations.embeddings.providers.sentence_transformer.HuggingFaceEmbeddings"
    )
    def test_default_config(self, mock_hf):
        from agent_platform.integrations.embeddings.providers.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        provider = SentenceTransformerEmbeddingProvider()
        assert provider._model == "sentence-transformers/all-MiniLM-L6-v2"
        assert isinstance(provider.config, SentenceTransformerConfig)

    @patch(
        "agent_platform.integrations.embeddings.providers.sentence_transformer.HuggingFaceEmbeddings"
    )
    def test_custom_model_and_config(self, mock_hf):
        from agent_platform.integrations.embeddings.providers.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        config = SentenceTransformerConfig(
            normalize=False, model_kwargs={"device": "cpu"}
        )
        provider = SentenceTransformerEmbeddingProvider(
            model="custom-model", config=config
        )
        assert provider._model == "custom-model"
        assert provider.config.normalize is False
        assert provider.config.model_kwargs == {"device": "cpu"}


class TestOpenAIEmbeddingConstructor:
    @patch("agent_platform.integrations.embeddings.providers.openai.OpenAIEmbeddings")
    def test_default_config(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from agent_platform.integrations.embeddings.providers.openai import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider()
        assert provider._model == "text-embedding-3-small"
        assert isinstance(provider.config, OpenAIEmbeddingConfig)

    @patch("agent_platform.integrations.embeddings.providers.openai.OpenAIEmbeddings")
    def test_custom_model(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from agent_platform.integrations.embeddings.providers.openai import (
            OpenAIEmbeddingProvider,
        )

        config = OpenAIEmbeddingConfig(dimensions=256, organization="org-1", timeout=30)
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-ada-002", config=config
        )
        assert provider._model == "text-embedding-ada-002"
        assert provider.config.dimensions == 256
        assert provider.config.organization == "org-1"
        assert provider.config.timeout == 30


class TestOllamaEmbeddingConstructor:
    @patch("agent_platform.integrations.embeddings.providers.ollama.OllamaEmbeddings")
    def test_default_config(self, mock_ollama):
        mock_ollama.return_value = MagicMock()
        from agent_platform.integrations.embeddings.providers.ollama import (
            OllamaEmbeddingProvider,
        )

        provider = OllamaEmbeddingProvider()
        assert provider._model == "nomic-embed-text"
        assert isinstance(provider.config, OllamaEmbeddingConfig)

    @patch("agent_platform.integrations.embeddings.providers.ollama.OllamaEmbeddings")
    def test_custom_model(self, mock_ollama):
        mock_ollama.return_value = MagicMock()
        from agent_platform.integrations.embeddings.providers.ollama import (
            OllamaEmbeddingProvider,
        )

        config = OllamaEmbeddingConfig(base_url="http://localhost:11434")
        provider = OllamaEmbeddingProvider(model="llama-embed", config=config)
        assert provider._model == "llama-embed"
        assert provider.config.base_url == "http://localhost:11434"


class TestMistralEmbeddingConstructor:
    @patch(
        "agent_platform.integrations.embeddings.providers.mistral.MistralAIEmbeddings"
    )
    def test_default_config(self, mock_mistral):
        mock_mistral.return_value = MagicMock()
        from agent_platform.integrations.embeddings.providers.mistral import (
            MistralEmbeddingProvider,
        )

        provider = MistralEmbeddingProvider()
        assert provider._model == "mistral-embed"
        assert isinstance(provider.config, MistralEmbeddingConfig)

    @patch(
        "agent_platform.integrations.embeddings.providers.mistral.MistralAIEmbeddings"
    )
    def test_custom_model(self, mock_mistral):
        mock_mistral.return_value = MagicMock()
        from agent_platform.integrations.embeddings.providers.mistral import (
            MistralEmbeddingProvider,
        )

        config = MistralEmbeddingConfig(timeout=60)
        provider = MistralEmbeddingProvider(model="custom-embed", config=config)
        assert provider._model == "custom-embed"
        assert provider.config.timeout == 60
