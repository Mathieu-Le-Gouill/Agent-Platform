from agent_platform.core.config import ProviderConfig


class OCRConfig(ProviderConfig):
    # Tesseract-style 3-letter code (e.g. "eng"); other providers reinterpret
    # or ignore this — see each provider's config. https://pypi.org/project/pytesseract
    language: str = "eng"
    # Filters extracted text below this confidence (0-100 or 0-1 depending on
    # provider's Score scale). https://cloud.google.com/vision/docs/ocr
    min_confidence: float = 0.0
