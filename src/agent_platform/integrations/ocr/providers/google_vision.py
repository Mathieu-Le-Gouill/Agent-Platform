from typing import Sequence
from uuid import UUID, uuid4
import asyncio

from google.cloud import vision

from agent_platform.integrations.ocr.base import BaseOCR
from agent_platform.integrations.ocr.config import OCRConfig, GoogleVisionConfig
from agent_platform.models.chunk import Chunk

from agent_platform.bridges.google.chunk import from_google_vision


class GoogleVisionOCR(BaseOCR):

    def __init__(
        self,
        config: GoogleVisionConfig | None = None,
    ) -> None:

        self._config = config or GoogleVisionConfig()
        self._client = self._build_client()


    def _build_client(self) -> vision.ImageAnnotatorClient:

        if self._config.credentials_path:
            return vision.ImageAnnotatorClient.from_service_account_file(
                self._config.credentials_path
            )

        return vision.ImageAnnotatorClient()


    async def extract(
        self,
        source: str,
        config: OCRConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[Chunk]:

        cfg = config or self._config
        doc_id = document_id or uuid4()

        image = await asyncio.to_thread(self._load_image, source)

        request = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=self._feature_type(cfg))],
        )

        response = await asyncio.to_thread(
            self._client.annotate_image,
            request=request,
        )

        if response.error.message:
            raise RuntimeError(response.error.message)

        return from_google_vision(
            response,
            document_id=doc_id,
            min_confidence=cfg.min_confidence,
        )


    @staticmethod
    def _feature_type(cfg: OCRConfig) -> vision.Feature.Type:

        name = getattr(cfg, "feature_type", "DOCUMENT_TEXT_DETECTION")

        return getattr(
            vision.Feature.Type,
            name,
            vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
        )


    @staticmethod
    def _load_image(source: str) -> vision.Image:

        if source.startswith("http://") or source.startswith("https://"):
            return vision.Image(source=vision.ImageSource(image_uri=source))

        with open(source, "rb") as f:
            content = f.read()

        return vision.Image(content=content)