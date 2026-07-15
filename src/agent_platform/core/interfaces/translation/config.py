from pydantic import BaseModel, ConfigDict


class TranslationConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    pass
