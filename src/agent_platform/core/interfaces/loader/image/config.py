from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.enums import ImageFormat


class ImageLoaderConfig(LoaderConfig):
    target_format: ImageFormat | None = None
    max_size: tuple[int, int] | None = None
