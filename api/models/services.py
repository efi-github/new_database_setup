from schemas import MODELS
from db import session, models
from db.models import EmbeddingModel, ReductionModel

def get_embedding_model(name, args):
    # look in database for model with name
    # if not found, create new model
    temp = MODELS[name](**args)
    db_model = session.query(EmbeddingModel).filter(EmbeddingModel.ModelName == temp.__str__()).first()
    if not db_model:
        # create model


def get_reduction_model(name, args):
    # look in database for model with name
    # if not found, create new model
    temp = MODELS[name](**args)
    db_model = session.query(ReductionModel).filter(ReductionModel.ModelName == temp.__str__()).first()
    if not db_model:
# create model