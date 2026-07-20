from agent_platform.core.interfaces.loader.config import LoaderConfig


class AudioLoaderConfig(LoaderConfig):
    # Resample audio to this rate (Hz) on load; None keeps the source rate.
    target_sample_rate: int | None = None
    # Truncate loaded audio to this duration, in seconds; None loads the full file.
    max_duration_sec: float | None = None
