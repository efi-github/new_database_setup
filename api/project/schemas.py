from pydantic import BaseModel
from typing import Optional



class ProjectCreate(BaseModel):
    ProjectName: str
    Description: Optional[str] = None


class ProjectBase(BaseModel):
    ProjectID: int
    ProjectName: str
    Description: str

class ProjectResponse(ProjectBase):
    class Config:
        orm_mode = True
