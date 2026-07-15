from pydantic import BaseModel, ConfigDict


class ImageGenConfig(BaseModel, frozen=True):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "dall-e-3"
