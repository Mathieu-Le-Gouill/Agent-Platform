from agent_platform.core.interfaces.loader.config import LoaderConfig


class VideoLoaderConfig(LoaderConfig):
    extract_audio: bool = False
    max_duration_sec: float | None = None
