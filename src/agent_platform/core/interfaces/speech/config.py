from pydantic import BaseModel, ConfigDict


class SpeechConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "nova-2"
