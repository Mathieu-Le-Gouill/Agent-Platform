from typing import Any

from integrations.embeddings.config import EmbeddingConfig


def to_sentence_transformers(
    config: EmbeddingConfig | None,
) -> dict[str, Any]:
    if config is None:
        return {
            "convert_to_tensor": True,
        }

    params: dict[str, Any] = {
        "convert_to_tensor": config.convert_to_tensor,
        "show_progress_bar": config.show_progress,
    }

    if config.batch_size is not None:
        params["batch_size"] = config.batch_size

    if config.device is not None:
        params["device"] = config.device

    if config.normalize:
        params["normalize_embeddings"] = True

    return params