from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db import session, models
from api.auth_utilities import get_current_user, TokenData
from api.project.schemas import ProjectResponse, ProjectCreate

router = APIRouter()

@router.get("/get_all", response_model=list[ProjectResponse])
def get_user_projects(db: Session = Depends(session.get_db), current_user: TokenData = Depends(get_current_user)):
    user = db.query(models.User).filter(models.User.Username == current_user.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.projects


@router.post("/create", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(session.get_db),
                   current_user: TokenData = Depends(get_current_user)):
    # Create a new Project instance
    new_project = models.Project(
        ProjectName=project.ProjectName,
        Description=project.Description,
        Username=current_user.username
    )

    db.add(new_project)
    db.commit()

    return new_project