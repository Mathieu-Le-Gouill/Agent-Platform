from agent_platform.core.interfaces.loader.config import LoaderConfig


class AudioLoaderConfig(LoaderConfig):
    target_sample_rate: int | None = None
    max_duration_sec: float | None = None
