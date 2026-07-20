from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_huggingface")

from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.core.errors import MissingCredentialError


class TestHuggingFaceToLangchain:
    def _creds(self):
        return HuggingFaceCredentials(api_key=SecretStr("hf_test"))

    def test_default_config(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig()
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert result["max_retries"] == 3
        assert result["temperature"] == 0.7

    def test_with_all_fields(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig(
            temperature=0.5,
            max_tokens=200,
            top_p=0.9,
            top_k=40,
            stop_sequences=["stop"],
            timeout=30.0,
            max_retries=5,
        )
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert result["temperature"] == 0.5
        assert result["max_new_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["stop"]
        assert result["timeout"] == 30
        assert result["max_retries"] == 5

    def test_temperature_zero(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig(temperature=0.0)
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert "temperature" not in result

    def test_max_tokens_zero(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig(max_tokens=0)
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert "max_new_tokens" not in result

    def test_new_fields_omitted_when_unset(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig()
        result = _to_langchain_hugging_face(cfg, self._creds())
        for key in ("repetition_penalty", "do_sample", "typical_p", "return_full_text"):
            assert key not in result

    def test_new_fields_forwarded_when_set(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig(
            repetition_penalty=1.1,
            do_sample=True,
            typical_p=0.9,
            return_full_text=False,
        )
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert result["repetition_penalty"] == 1.1
        assert result["do_sample"] is True
        assert result["typical_p"] == 0.9
        assert result["return_full_text"] is False

    def test_do_sample_false_is_forwarded(self):
        # Regression: `if config.do_sample:` would drop an explicit False;
        # must use `is not None` gating.
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig(do_sample=False)
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert result["do_sample"] is False

    def test_return_full_text_false_is_forwarded(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            _to_langchain_hugging_face,
        )

        cfg = HuggingFaceGenerationConfig(return_full_text=False)
        result = _to_langchain_hugging_face(cfg, self._creds())
        assert result["return_full_text"] is False


class TestHuggingFaceLLMConstruction:
    def test_construct(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            HuggingFaceLLM,
        )

        provider = HuggingFaceLLM(HuggingFaceCredentials(api_key=SecretStr("hf_test")))
        assert provider._credentials.api_key.get_secret_value() == "hf_test"

    def test_default_config(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            HuggingFaceLLM,
        )

        provider = HuggingFaceLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, HuggingFaceGenerationConfig)
        assert cfg.repo_id == "deepseek-ai/DeepSeek-R1-0528"
        assert cfg.task == "text-generation"
        assert cfg.provider == "auto"


class TestHuggingFaceMissingCredential:
    def test_missing_api_key_raises(self):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            HuggingFaceLLM,
        )

        provider = HuggingFaceLLM()
        provider._credentials = HuggingFaceCredentials()
        cfg = HuggingFaceGenerationConfig(repo_id="test/model")
        with pytest.raises(
            MissingCredentialError,
            match="Hugging Face Hub API token is required",
        ):
            provider._client(cfg)


class TestHuggingFaceClient:
    @patch("agent_platform.integrations.llm.huggingface.huggingface.ChatHuggingFace")
    @patch("agent_platform.integrations.llm.huggingface.huggingface.HuggingFaceEndpoint")
    def test_client_creation(self, mock_endpoint, mock_chat):
        from agent_platform.integrations.llm.huggingface.huggingface import (
            HuggingFaceLLM,
        )

        mock_endpoint_instance = MagicMock()
        mock_endpoint.return_value = mock_endpoint_instance
        mock_chat_instance = MagicMock()
        mock_chat.return_value = mock_chat_instance

        provider = HuggingFaceLLM(HuggingFaceCredentials(api_key=SecretStr("hf_test")))
        config = HuggingFaceGenerationConfig(
            repo_id="test/model", task="text-generation"
        )
        client = provider._client(config)

        mock_endpoint.assert_called_once()
        mock_chat.assert_called_once_with(llm=mock_endpoint_instance)
        assert client is mock_chat_instance
