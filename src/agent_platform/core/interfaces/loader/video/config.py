from agent_platform.core.interfaces.loader.config import LoaderConfig


class VideoLoaderConfig(LoaderConfig):
    # If True, also extract the video's audio track on load.
    extract_audio: bool = False
    # Truncate loaded video to this duration, in seconds; None loads the full file.
    max_duration_sec: float | None = None
