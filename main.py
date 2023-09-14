from fastapi.security import OAuth2PasswordBearer

from db.models import *
from db.session import engine
from db.base import Base
from fastapi import FastAPI
from api.user.router import router as user_router
from api.project.router import router as project_router
from api.dataset.router import router as dataset_router
from api.view.router import router as view_router

import uvicorn


#app = FastAPI()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/user", tags=["User"])
app.include_router(project_router, prefix="/project", tags=["Project"])
app.include_router(dataset_router, prefix="/project/{project_id}/dataset", tags=["Dataset"])
app.include_router(view_router, prefix="/project/{project_id}/view", tags=["View"])
"""
app.include_router(annotation.router, prefix="/annotation", tags=["Annotation"])
app.include_router(dataset.router, prefix="/dataset", tags=["Dataset"])
app.include_router(embedding.router, prefix="/embedding", tags=["Embedding"])
app.include_router(position.router, prefix="/position", tags=["Position"])

app.include_router(sentence.router, prefix="/sentence", tags=["Sentence"])

app.include_router(view.router, prefix="/view", tags=["View"])
"""

def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    init_db()




