from pydantic import BaseModel, ConfigDict


class LoaderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
