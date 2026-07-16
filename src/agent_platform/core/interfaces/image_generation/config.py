from agent_platform.core.schemas.config import ProviderConfig


class ImageGenConfig(ProviderConfig):
    model: str = "dall-e-3"
