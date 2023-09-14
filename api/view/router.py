import logging
import pickle

import numpy as np
from fastapi import APIRouter, Body, Depends
from sqlalchemy import not_, and_, exists
from sqlalchemy.orm import Session, aliased

from api.auth_utilities import TokenData, get_current_user
from api.view.schemas import ModelDefinition
from db import session, models
from api.models.schemas import MODELS
from db.models import Embedding, Segment, Sentence, Dataset, User, Project, Position, CombinedModel
from db.session import get_db, SessionLocal
import time

from utilities import Timer

router = APIRouter()
model_directory = "/home/efi/PycharmProjects/new_database_setup/api/models/model_files"


def get_model(model_data: ModelDefinition, project_id, username, db: Session = Depends(get_db), name_prefix=""):
    def get_model_db(model_name, project_id):
        print(model_name, project_id)
        return (
            db.query(models.CombinedModel)
            .join(models.Project, models.CombinedModel.ProjectID == models.Project.ProjectID)
            .filter(models.CombinedModel.ProjectID == project_id)
            .filter(models.CombinedModel.ModelName == model_name)
            .first()
        )

    name = model_data.name
    args = model_data.args
    model = MODELS[name](fitted=False, arguments=args)
    model_data.name = name_prefix + model.__str__()
    db_model = get_model_db(model_data.name, project_id)
    if not db_model:
        save_model(model_data, model, project_id, username, db)
        db_model = get_model_db(model_data.name, project_id)
    with open(f"{model_directory}/{db_model.ModelFile}", "rb") as f:
        model = pickle.load(f)
    return model, db_model


def update_model(model_data, model, project_id, user):
    filename = f"{user}_{project_id}_{model_data.name}.pkl".replace("-", "_").replace(" ", "_").replace("/", "_")
    with open(f"{model_directory}/{filename}", "wb") as f:
        pickle.dump(model, f)


def save_model(model_data, model, project_id, user, db: Session = Depends(get_db)):
    filename = f"{user}_{project_id}_{model_data.name}.pkl".replace("-", "_").replace(" ", "_").replace("/", "_")
    with open(f"{model_directory}/{filename}", "wb") as f:
        pickle.dump(model, f)
    db_model = models.CombinedModel(
        ProjectID=project_id,
        ModelName=model_data.name,
        ModelFile=filename,

    )
    db.add(db_model)
    db.commit()


@router.post("/create")
def create_view(
        project_id: int,
        embedding_model_data: ModelDefinition = Body(...),
        reduction_model_data: ModelDefinition = Body(...),
        db: Session = Depends(get_db),
        current_user: TokenData = Depends(get_current_user)
):
    result = (
        db.query(models.Project)
        .filter(
            models.Project.ProjectID == project_id
        )
        .all()
    )
    if len(result) == 0 or result[0].user.Username != current_user.username:
        return {"error": "Project not found"}
    # initialize, so get the models necessary
    with Timer("Get embedding and reduction models"):
        embedding_model, db_e = get_model(embedding_model_data, project_id, current_user.username, db)
        reduction_model, db_r = get_model(reduction_model_data, project_id, current_user.username, db,
                                          name_prefix=db_e.ModelName + "_")

    # get any segments that don't have embeddings
    # get both the segments and their corresponding sentence
    with Timer("Get segments and sentences to embed"):
        subquery = exists().where(
            and_(
                Embedding.SegmentID == Segment.SegmentID,
                Embedding.ModelID == db_e.ModelID
            )
        )

        # Initialize the query for Segments and corresponding Sentences
        segments_and_sentences = db.query(Segment, Sentence). \
            join(Sentence, Sentence.SentenceID == Segment.SentenceID). \
            join(Dataset, Dataset.DatasetID == Sentence.DatasetID). \
            join(Project, Project.ProjectID == Dataset.ProjectID). \
            filter(Project.ProjectID == project_id). \
            filter(not_(subquery)). \
            all()
        segments, sentences = [], []
        logging.log(logging.INFO, f"Segments and Sentences: #{len(segments_and_sentences)}")
        if len(segments_and_sentences) > 0:
            segments, sentences = zip(*segments_and_sentences)
        del segments_and_sentences
    if len(segments) > 0:
        with Timer("Generate Embeddings"):
            if not hasattr(embedding_model, "fitted") or not embedding_model.fitted:
                embedding_model.fit(segments, sentences)
            embedding_values = embedding_model.transform(segments, sentences)
        with Timer("Insert Embeddings"):
            embedding_mappings = [
                {
                    "SegmentID": segment.SegmentID,
                    "ModelID": db_e.ModelID,
                    "EmbeddingValues": pickle.dumps(embedding_value)
                }
                for embedding_value, segment in zip(embedding_values, segments)
            ]

            # Bulk insert embeddings
            db.bulk_insert_mappings(models.Embedding, embedding_mappings)
            db.commit()
    del segments
    del sentences
    with Timer("Get embeddings to reduce"):
        subquery = (
            db.query(models.Position.EmbeddingID)
            .filter(models.Position.ModelID == db_r.ModelID)
            .subquery()
        )

        # Main query to find embeddings
        embeddings_todo = db.query(Embedding). \
            join(CombinedModel, CombinedModel.ModelID == Embedding.ModelID). \
            join(Project, Project.ProjectID == CombinedModel.ProjectID). \
            filter(
            and_(
                Project.ProjectID == project_id,
                CombinedModel.ModelID == db_e.ModelID
            )
        ). \
            filter(
            not_(
                exists().where(
                    and_(
                        Position.EmbeddingID == Embedding.EmbeddingID,
                        Position.ModelID == db_r.ModelID
                    )
                )
            )
        ).all()
        logging.log(logging.INFO, f"Embeddings: #{len(embeddings_todo)}")
        embeddings_arrays = np.array([])
        if len(embeddings_todo) > 0:
            embeddings_arrays = np.stack([pickle.loads(embedding.EmbeddingValues) for embedding in embeddings_todo])
    if len(embeddings_arrays) > 0:
        with Timer("Generate Positions"):
            if not hasattr(reduction_model, "fitted") or not reduction_model.fitted:
                reduction_model.fit(embeddings_arrays)
            position_values = reduction_model.transform(embeddings_arrays)
        with Timer("Insert Positions"):
            position_mappings = [
                {
                    "EmbeddingID": embedding.EmbeddingID,
                    "ModelID": db_r.ModelID,
                    "Posx": float(position_value[0]),
                    "Posy": float(position_value[1])
                }
                for position_value, embedding in zip(position_values, embeddings_todo)
            ]
            db.bulk_insert_mappings(models.Position, position_mappings)
            db.commit()

    with Timer("Create View"):
        new_view = models.View(
            ProjectID=project_id,
            EmbeddingModelID=db_e.ModelID,
            ReductionModelID=db_r.ModelID
        )
        db.add(new_view)
        db.commit()
    update_model(embedding_model_data, embedding_model, project_id, current_user.username)
    update_model(reduction_model_data, reduction_model, project_id, current_user.username)
    return_dictionary = {
        "ViewID": new_view.ViewID,
        "ProjectID": new_view.ProjectID,
        "EmbeddingModelID": new_view.EmbeddingModelID,
        "ReductionModelID": new_view.ReductionModelID
    }
    db.close()
    del embeddings_todo
    del embedding_model
    del reduction_model
    return return_dictionary


@router.post("/create1")
def create_view1(
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
        PositionAlias.PositionID == None
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
    res = create_view(1, ModelDefinition(name="bert", args={
        "pretrained_model_name_or_path": "dbmdz/bert-large-cased-finetuned-conll03-english"}),
                      ModelDefinition(name="umap", args={"n_jobs": -1}), db=db,
                      current_user=TokenData(username="string"))
