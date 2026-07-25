from agent_platform.components.llm_classifier.config import (
    AnthropicLLMClassifierConfig,
    ClassificationExample,
    ClassificationMode,
    HuggingFaceLLMClassifierConfig,
    LLMClassifierConfig,
    MistralLLMClassifierConfig,
    OllamaLLMClassifierConfig,
    OpenAILLMClassifierConfig,
)


class TestLLMClassifierConfig:
    def test_defaults(self):
        config = LLMClassifierConfig()
        assert config.classification_mode == ClassificationMode.TEXT_CLASSIFICATION
        assert config.multi_label is False
        assert config.unknown_label == "unknown"
        assert config.examples == []

    def test_construction(self):
        config = LLMClassifierConfig(
            classification_mode=ClassificationMode.ZERO_SHOT,
            multi_label=True,
            threshold=0.5,
            unknown_label="none",
            return_confidence=True,
            examples=[ClassificationExample(text="hi", label="greeting")],
        )
        assert config.classification_mode == ClassificationMode.ZERO_SHOT
        assert config.multi_label is True
        assert config.threshold == 0.5
        assert config.examples[0].label == "greeting"


class TestProviderConfigs:
    def test_anthropic_config_mixes_generation_config(self):
        config = AnthropicLLMClassifierConfig(model="claude-3-opus")
        assert config.model == "claude-3-opus"
        assert config.classification_mode == ClassificationMode.TEXT_CLASSIFICATION

    def test_openai_config_mixes_generation_config(self):
        config = OpenAILLMClassifierConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_mistral_config_mixes_generation_config(self):
        config = MistralLLMClassifierConfig(model="mistral-large")
        assert config.model == "mistral-large"

    def test_ollama_config_has_default_model(self):
        config = OllamaLLMClassifierConfig()
        assert config.model == "mikgr/doctype-classifier-vl"

    def test_huggingface_config_has_default_model(self):
        config = HuggingFaceLLMClassifierConfig()
        assert config.model == "distilbert-base-uncased-finetuned-sst-2-english"
