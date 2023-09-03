import json
import shutil
from fastapi import APIRouter, Depends, UploadFile, HTTPException, File, Body
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import session, models
from api.auth_utilities import get_current_user, TokenData
from api.dataset.schemas import DatasetCreate, DatasetTextOptions
from api.dataset.service import text_to_json, add_data_to_db

from pprint import pprint

router = APIRouter()


@router.post("/upload")
async def upload_dataset(
        project_id: int,
        dataset_name: str,
        split: str = "\t",
        sentence_split: str = "\n\n",
        word_idx: int = 1,
        label_idx: int = 0,
        label_split: str = "None",
        type: str = "B-I-O",
        file: UploadFile = File(...),
        db: Session = Depends(session.get_db),
        current_user: TokenData = Depends(get_current_user)
):
    # Check if the project exists and belongs to the user
    project = db.query(models.Project).filter(models.Project.ProjectID == project_id,
                                              models.Project.Username == current_user.username).first()
    #print(sentence_split.encode().decode('unicode_escape'))
    options = DatasetTextOptions(split=split.encode().decode('unicode_escape'), sentence_split=sentence_split.encode().decode('unicode_escape'), type = type.encode().decode('unicode_escape'), word_idx=word_idx, label_idx=label_idx, label_split=label_split.encode().decode('unicode_escape'))

    if not project:
        raise HTTPException(status_code=404, detail="Project not found or you don't have permission to access it.")

    #print(file.content_type)
    file_content = await file.read()
    file_content = file_content.decode("utf-8")
    #print(file.content_type)
    #print(file.filename)
    # 1. By MIME Type
    extension = file.filename.split(".")[-1]
    if extension == "txt":
        file_type = "PlainText"
        temp_dictionary = text_to_json(file_content, options)
        #pprint(temp_dictionary)
    elif extension == "json":
        file_type = "JSON"
        temp_dictionary = json.loads(file_content)
    else:
        return {"error": "Unsupported file type"}
    # Read the file content

    add_data_to_db(current_user.username, project_id, dataset_name, temp_dictionary, db)

    # Return a response or the processed data as desired
    return {"filename": file.filename, "content_length": len(file_content)}

