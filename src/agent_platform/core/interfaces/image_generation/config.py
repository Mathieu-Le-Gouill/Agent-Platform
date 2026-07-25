from agent_platform.core.config import ProviderConfig


class ImageGenConfig(ProviderConfig):
    # Default is OpenAI-shaped; each provider overrides with its own model identifier.
    # https://platform.openai.com/docs/api-reference/images/create#images-create-model
    model: str = "dall-e-3"
