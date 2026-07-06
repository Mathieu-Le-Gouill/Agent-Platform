from pydantic import SecretStr

from agent_platform.integrations.reranking.config import (
    RerankerConfig,
    CohereRerankerConfig,
    JinaRerankerConfig,
    HuggingFaceRerankerConfig,
)


def test_reranker_config_defaults():
    cfg = RerankerConfig()
    assert cfg.top_k is None
    assert cfg.return_scores is False
    assert cfg.batch_size == 32
    assert cfg.normalize_scores is True


def test_reranker_config_custom():
    cfg = RerankerConfig(
        top_k=5, return_scores=True, batch_size=16, normalize_scores=False
    )
    assert cfg.top_k == 5
    assert cfg.return_scores is True
    assert cfg.batch_size == 16
    assert cfg.normalize_scores is False


def test_cohere_reranker_config_defaults():
    cfg = CohereRerankerConfig()
    assert isinstance(cfg, RerankerConfig)
    assert cfg.top_k is None
    assert cfg.batch_size == 32


def test_cohere_reranker_config_explicit_api_key():
    cfg = CohereRerankerConfig(api_key=SecretStr("test-key"))
    assert cfg.api_key.get_secret_value() == "test-key"


def test_jina_reranker_config_defaults():
    cfg = JinaRerankerConfig()
    assert isinstance(cfg, RerankerConfig)


def test_jina_reranker_config_explicit_api_key():
    cfg = JinaRerankerConfig(api_key=SecretStr("jina-key"))
    assert cfg.api_key.get_secret_value() == "jina-key"


def test_huggingface_reranker_config_defaults():
    cfg = HuggingFaceRerankerConfig()
    assert isinstance(cfg, RerankerConfig)
    assert cfg.device is None


def test_huggingface_reranker_config_device():
    cfg = HuggingFaceRerankerConfig(device="cuda:0")
    assert cfg.device == "cuda:0"


def test_configs_are_frozen():
    cfg = RerankerConfig(top_k=3)
    try:
        cfg.top_k = 5  # type: ignore[misc]
        assert False, "expected FrozenInstanceError"
    except AttributeError:
        pass
