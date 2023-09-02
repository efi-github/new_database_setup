from pydantic import BaseModel

class DatasetCreate(BaseModel):
    name: str
    file_path: str
