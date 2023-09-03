from sqlalchemy.orm import aliased
from sqlalchemy import and_, not_

from db.models import Segment, Embedding, Sentence, Dataset, Project, User
from db.session import SessionLocal
session = SessionLocal()
# Create aliases for tables you will refer to multiple times in your query
SegmentAlias = aliased(Segment)
EmbeddingAlias = aliased(Embedding)

query = session.query(SegmentAlias).join(
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
        EmbeddingAlias.ModelName == "BertEmbeddingModel({'pretrained_model_name_or_path': 'bert-base-uncased'})"
    )
).filter(
    User.Username == 'string',
    Project.ProjectID == 1,
    EmbeddingAlias.EmbeddingID == None  # This filters out the segments that have embeddings
).all()

# Display the results
for segment in query:
    print(segment.SegmentID, segment.Text)
    for embed in segment.embedding:
        print(embed.EmbeddingID, embed.ModelName)