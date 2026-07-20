from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.enums import ImageFormat


class ImageLoaderConfig(LoaderConfig):
    # Convert loaded images to this format; None keeps the source format.
    target_format: ImageFormat | None = None
    # Maximum (width, height) to downscale loaded images to; None keeps the source size.
    max_size: tuple[int, int] | None = None
