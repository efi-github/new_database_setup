from sqlalchemy.orm import aliased
from sqlalchemy import and_, not_

from db.models import Segment, Embedding, Sentence, Dataset, Project, User, Position
from db.session import SessionLocal
session = SessionLocal()
# Create aliases for tables you will refer to multiple times in your query

EmbeddingAlias = aliased(Embedding)
PositionAlias = aliased(Position)

query = session.query(EmbeddingAlias).join(
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
        PositionAlias.ModelName == 'umap'
    )
).filter(
    User.Username == 'string',
    Project.ProjectID == 1,
    PositionAlias.PositionID == None  # This filters out the embeddings that already have positions
).all()

# Display the results
for embedding in query:
    print(embedding.EmbeddingID, embedding.ModelName)