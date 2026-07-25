from pydantic import ValidationError

from agent_platform.core.interfaces.classification.config import ClassificationConfig


def test_classification_config_defaults():
    cfg = ClassificationConfig()
    assert cfg.multi_label is False


def test_classification_config_construction():
    cfg = ClassificationConfig(multi_label=True)
    assert cfg.multi_label is True


def test_classification_config_is_frozen():
    cfg = ClassificationConfig()
    try:
        cfg.multi_label = True
        assert False, "expected ValidationError"
    except ValidationError:
        pass
