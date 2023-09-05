from sqlalchemy.orm import aliased
from sqlalchemy import and_, not_, exists

from db import models
from db.models import Segment, Embedding, Sentence, Dataset, Project, User, Position, CombinedModel
from db.session import SessionLocal
db = SessionLocal()
# Create aliases for tables you will refer to multiple times in your query

EmbeddingAlias = aliased(Embedding)
PositionAlias = aliased(Position)

db_e_ModelID = 1
db_r_ModelID = 1
project_id = 1

subquery = (
            db.query(models.Position.EmbeddingID)
            .filter(models.Position.ModelID == db_r_ModelID)
            .subquery()
        )
query = db.query(Embedding).\
            join(CombinedModel, CombinedModel.ModelID == Embedding.ModelID).\
            join(Project, Project.ProjectID == CombinedModel.ProjectID).all()
"""filter(
                and_(
                    Project.ProjectID == project_id,
                    CombinedModel.ModelID == db_e_ModelID
                )
            ).all()"""
"""filter(
    not_(
        exists().where(
            and_(
                Position.EmbeddingID == Embedding.EmbeddingID,
                Position.ModelID == db_r_ModelID
            )
        )
    )
).all()"""

# Display the results
for embedding in query:
    print(embedding.EmbeddingID, embedding.ModelName)