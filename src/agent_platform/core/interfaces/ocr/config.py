from agent_platform.core.config import ProviderConfig


class OCRConfig(ProviderConfig):
    # Filters extracted text below this confidence (0-100 or 0-1 depending on
    # provider's Score scale). https://cloud.google.com/vision/docs/ocr
    min_confidence: float = 0.0
