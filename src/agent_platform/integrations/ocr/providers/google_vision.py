from typing import Sequence
from uuid import UUID, uuid4
import asyncio

from google.cloud import vision

from agent_platform.integrations.ocr.base import BaseOCR
from agent_platform.integrations.ocr.config import GoogleVisionConfig
from agent_platform.models.chunk import TextChunk


def _from_google_vision(
    response, document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                words = paragraph.words
                if not words:
                    continue

                text = " ".join(
                    "".join(symbol.text for symbol in word.symbols) for word in words
                )
                confs = [
                    symbol.confidence
                    for word in words
                    for symbol in word.symbols
                    if symbol.confidence
                ]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                if avg_conf < min_confidence:
                    continue

                chunks.append(
                    TextChunk(
                        id=uuid4(),
                        document_id=document_id,
                        text=text.strip(),
                        metadata={
                            "confidence": avg_conf,
                            "page": page.page_number,
                        },
                    )
                )

    return chunks


class GoogleVisionOCR(BaseOCR[GoogleVisionConfig]):
    def __init__(self, config: GoogleVisionConfig | None = None) -> None:
        self._config = config or GoogleVisionConfig()
        self._client = self._build_client()

    def _build_client(self) -> vision.ImageAnnotatorClient:
        if self._config.credentials_path:
            return vision.ImageAnnotatorClient.from_service_account_file(
                self._config.credentials_path,
            )
        return vision.ImageAnnotatorClient()

    async def extract(
        self,
        source: str,
        config: GoogleVisionConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        cfg = config or self._config
        doc_id = document_id or uuid4()
        image = await asyncio.to_thread(self._load_image, source)

        request = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=_feature_type(cfg.feature_type))],
        )

        response = await asyncio.to_thread(
            self._client.annotate_image,
            request=request,
        )

        if response.error.message:
            raise RuntimeError(response.error.message)

        return _from_google_vision(
            response, document_id=doc_id, min_confidence=cfg.min_confidence
        )

    @staticmethod
    def _load_image(source: str) -> vision.Image:
        if source.startswith("http://") or source.startswith("https://"):
            return vision.Image(source=vision.ImageSource(image_uri=source))
        with open(source, "rb") as f:
            content = f.read()
        return vision.Image(content=content)


def _feature_type(name: str) -> vision.Feature.Type:
    return getattr(
        vision.Feature.Type,
        name,
        vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
    )
