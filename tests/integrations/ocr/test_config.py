from agent_platform.core.interfaces.ocr.config import OCRConfig
from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig
from agent_platform.integrations.ocr.tesseract.config import TesseractConfig


def test_tesseract_config_inherits_base_defaults():

    cfg = TesseractConfig()

    assert isinstance(cfg, OCRConfig)
    assert cfg.language == "eng"
    assert cfg.min_confidence == 0.0
    assert cfg.psm == 3
    assert cfg.oem == 3


def test_google_vision_config_overrides():

    cfg = GoogleVisionConfig(language="fr", min_confidence=0.6)

    assert cfg.language == "fr"
    assert cfg.min_confidence == 0.6
    assert cfg.feature_type == "DOCUMENT_TEXT_DETECTION"


from pydantic import ValidationError


def test_configs_are_frozen():

    cfg = AWSTextractConfig()

    try:
        cfg.region_name = "eu-west-1"  # type: ignore[misc]
        assert False, "expected ValidationError"
    except ValidationError:
        pass
