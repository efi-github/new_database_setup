import pickle

from fastapi import Depends

from db.session import get_db
from schemas import MODELS
from db import session, models
from db.models import EmbeddingModel, ReductionModel

"""def get_embedding_model(model, db = Depends(get_db)):
    # look in database for model with name
    # if not found, create new model
    db_model = session.query(EmbeddingModel).filter(EmbeddingModel.ModelName == model.name).first()
    if db_model:
        if 
        model = pickle.loads(db_model.ModelPickle)
    
    
    if not db_model:
        db_model = EmbeddingModel(
            ModelName=model.name,
            ModelDescription="Description here...", 
            ModelPickle = pickle.dumps(model)
        )
        session.add(db_model)
        session.commit()

    return db_model


def get_reduction_model(name, args, db = Depends(get_db))
    # look in database for model with name
    # if not found, create new model
    temp = MODELS[name](**args)
    db_model = session.query(ReductionModel).filter(ReductionModel.ModelName == temp.__str__()).first()
    if not db_model:
        db_model = ReductionModel(
            ModelName=temp.__str__(),
            ModelDescription="Description here...",
            ModelPickle=pickle.dumps(temp)
        )
        session.add(db_model)
        session.commit()

    return db_model"""