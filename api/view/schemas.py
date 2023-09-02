from pydantic import BaseModel


class ModelDefinition(BaseModel):
    name: str
    args: dict