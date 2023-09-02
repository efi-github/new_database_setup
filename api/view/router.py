from fastapi import APIRouter, Body

from api.view.schemas import ModelDefinition
from db import session, models
from api.models.schemas import MODELS

router = APIRouter()

@router.post("/create")
def create_view(
    embedding_model_data: ModelDefinition = Body(...),
    reduction_model_data: ModelDefinition = Body(...)
):
    # Your logic here to create the models using `embedding_model` and `reduction_model`
    embedding_name = embedding_model_data.name
    embedding_args = embedding_model_data.args
    embedding_model = MODELS[embedding_name](**embedding_args)
    # check if the name of any EmbeddingModel in the db has name embedding_model.__str__()
    db_e_model = session.query(models.EmbeddingModel).filter(models.EmbeddingModel.Name == embedding_model.__str__()).first()
    if not db_e_model:
        EmbeddingModel = models.EmbeddingModel(Name=embedding_model.__str__(), Args=embedding_args)


        reduction_name = reduction_model_data.name
        reduction_args = reduction_model_data.args
        reduction_model = MODELS[reduction_name](**reduction_args)

    # Do something with the models (initialize, train, etc.)
    # ...

    return {
        "message": "Models created successfully!",
        "embedding_model": {
            "name": embedding_name,
            "args": embedding_args
        },
        "reduction_model": {
            "name": reduction_name,
            "args": reduction_args
        }
    }