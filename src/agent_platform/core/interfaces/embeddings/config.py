from agent_platform.core.schemas.config import ProviderConfig


class EmbeddingConfig(ProviderConfig):
    # Provider-specific model identifier; each provider overrides the default.
    model: str = ""
    # Local input-list batching size used by `components/embedder.py` before
    # calling the provider (see `chunked(input, batch_size)`), independent of
    # any provider-side request-batching knob.
    batch_size: int = 32
    # Output embedding vector size. Only a subset of providers/models support
    # truncating dimensions (e.g. OpenAI text-embedding-3-*); not all clients
    # accept this field, see each provider's mapper for gating.
    dimensions: int | None = None
    # Per-request timeout in seconds. Not every client exposes a compatible
    # top-level field; see each provider's mapper for gating/adaptation.
    timeout: float | None = None
