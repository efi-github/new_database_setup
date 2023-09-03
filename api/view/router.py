import pickle

import numpy as np
from fastapi import APIRouter, Body, Depends
from sqlalchemy import not_, and_
from sqlalchemy.orm import Session, aliased

from api.auth_utilities import TokenData, get_current_user
from api.view.schemas import ModelDefinition
from db import session, models
from api.models.schemas import MODELS
from db.models import Embedding, Segment, Sentence, Dataset, User, Project, Position
from db.session import get_db, SessionLocal
import time

router = APIRouter()

@router.post("/create")
def create_view(
    project_id: int,
    embedding_model_data: ModelDefinition = Body(...),
    reduction_model_data: ModelDefinition = Body(...),
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    origin_time = time.time()
    current_time = origin_time

    # Get or create embedding model
    embedding_name = embedding_model_data.name
    embedding_args = embedding_model_data.args
    embedding_model = MODELS[embedding_name](**embedding_args)
    embedding_model.name = embedding_model.__str__()

    SegmentAlias = aliased(Segment)
    EmbeddingAlias = aliased(Embedding)

    segments_and_sentences = db.query(SegmentAlias, Sentence).join(
        Sentence, Sentence.SentenceID == SegmentAlias.SentenceID
    ).join(
        Dataset, Dataset.DatasetID == Sentence.DatasetID
    ).join(
        Project, Project.ProjectID == Dataset.ProjectID
    ).join(
        User, User.Username == Project.Username
    ).outerjoin(
        EmbeddingAlias, and_(
            EmbeddingAlias.SegmentID == SegmentAlias.SegmentID,
            EmbeddingAlias.ModelName == embedding_model.name
        )
    ).filter(
        User.Username == current_user.username,
        Project.ProjectID == project_id,
        EmbeddingAlias.EmbeddingID == None  # This filters out the segments that have embeddings
    ).all()
    if len(segments_and_sentences) > 0:
        segments, sentences = zip(*segments_and_sentences)
    else:
        segments, sentences = [], []

    last_time = current_time
    current_time = time.time()
    print(f"Querying the database for Embedding sentence, segment pairs took {current_time - last_time} seconds.")

    db_e_model = db.query(models.EmbeddingModel).filter(models.EmbeddingModel.ModelName == embedding_model.name).first()
    if db_e_model:
        embedding_model = pickle.loads(db_e_model.ModelPickle)
    else:
        embedding_model.fit(segments_and_sentences)
        db_e_model = models.EmbeddingModel(
            ModelName=embedding_model.name,
            ModelPickle=pickle.dumps(embedding_model)
        )
        db.add(db_e_model)
        db.commit()
        db.flush()

    last_time = current_time
    current_time = time.time()
    print(f"Prepareing Embedding segments took {current_time - last_time} seconds.")
    embedding_values = embedding_model.transform(segments, sentences)
    last_time = current_time
    current_time = time.time()
    print(f"Calculating Embeddings took {current_time - last_time} seconds.")
    print(f"Embedding values shape: {len(embedding_values)}")
    print(f"Segments: {len(segments)}")
    embedding_mappings = [
        {
            "SegmentID": segment.SegmentID,
            "ModelName": embedding_model.name,
            "EmbeddingValues": pickle.dumps(embedding_value)
        }
        for embedding_value, segment in zip(embedding_values, segments)
    ]

    # Bulk insert embeddings
    db.bulk_insert_mappings(models.Embedding, embedding_mappings)

    db.commit()
    last_time = current_time
    current_time = time.time()
    print(f"Inserting Embeddings took {current_time - last_time} seconds.")

    # Get or create reduction model
    reduction_name = reduction_model_data.name
    reduction_args = reduction_model_data.args
    reduction_model = MODELS[reduction_name](**reduction_args)
    reduction_model.name = embedding_model.__str__() + " " + reduction_model.__str__()

    EmbeddingAlias = aliased(Embedding)
    PositionAlias = aliased(Position)

    # Find embeddings needing positions
    embeddings_todo = db.query(EmbeddingAlias).join(
        Segment, Segment.SegmentID == EmbeddingAlias.SegmentID
    ).join(
        Sentence, Sentence.SentenceID == Segment.SentenceID
    ).join(
        Dataset, Dataset.DatasetID == Sentence.DatasetID
    ).join(
        Project, Project.ProjectID == Dataset.ProjectID
    ).join(
        User, User.Username == Project.Username
    ).outerjoin(
        PositionAlias, and_(
            PositionAlias.EmbeddingID == EmbeddingAlias.EmbeddingID,
            PositionAlias.ModelName == reduction_model.name
        )
    ).filter(
        User.Username == current_user.username,
        Project.ProjectID == project_id,
        PositionAlias.PositionID == None  # This filters out the embeddings that already have positions
    ).all()
    print("Embeddings TODO:", len(embeddings_todo))
    last_time = current_time
    current_time = time.time()
    print(f"Querying the database for Position segments took {current_time - last_time} seconds.")

    embedding_arrays = [pickle.loads(embedding.EmbeddingValues) for embedding in embeddings_todo]
    if len(embedding_arrays) != 0:
        numpy_embeddings = np.stack(embedding_arrays)
    else:
        numpy_embeddings = np.array([])
    db_r_model = db.query(models.ReductionModel).filter(models.ReductionModel.ModelName == reduction_model.name).first()
    if db_r_model:
        reduction_model = pickle.loads(db_r_model.ModelPickle)
    else:
        reduction_model.fit(numpy_embeddings)
        db_r_model = models.ReductionModel(
            ModelName=reduction_model.name,
            ModelPickle=pickle.dumps(reduction_model)
        )
        db.add(db_r_model)
        db.commit()
        db.flush()

    last_time = current_time
    current_time = time.time()
    print(f"Prepareing Position segments took {current_time - last_time} seconds.")
    position_values = reduction_model.transform(numpy_embeddings)
    last_time = current_time
    current_time = time.time()
    print(f"Calculating Positions took {current_time - last_time} seconds.")
    # Create positions for those embeddings
    position_mappings = [
        {
            "EmbeddingID": embedding.EmbeddingID,
            "ModelName": reduction_model.name,
            "Posx": float(position_value[0]),
            "Posy": float(position_value[1])
        }
        for position_value, embedding in zip(position_values, embeddings_todo)
    ]

    # Bulk insert positions
    db.bulk_insert_mappings(models.Position, position_mappings)

    db.commit()
    last_time = current_time
    current_time = time.time()
    print(f"Inserting Positions took {current_time - last_time} seconds.")

    new_view = models.View(
        ProjectID=project_id,
        EmbeddingModelName=db_e_model.ModelName,
        ReductionModelName=db_r_model.ModelName
    )
    db.add(new_view)

    # Commit changes to the database
    db.commit()
    db.refresh(new_view)

    print(f"Total time taken: {time.time() - origin_time} seconds.")
    return {
        "ViewID": new_view.ViewID,
        "ProjectID": new_view.ProjectID,
        "EmbeddingModelName": new_view.EmbeddingModelName,
        "ReductionModelName": new_view.ReductionModelName
    }



if __name__ == "__main__":
    db = SessionLocal()
    res = create_view(4, ModelDefinition(name="bert", args={}), ModelDefinition(name="semisupervised_umap", args={}), db=db, current_user=TokenData(username="string"))