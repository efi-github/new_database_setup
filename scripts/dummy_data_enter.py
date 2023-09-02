from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal


# Create a new session
session = SessionLocal()

# Insert User
new_user = User(Username="john_doe", Email="john@example.com", Password="securepassword123")
session.add(new_user)

# Insert Project
new_project = Project(Username="john_doe", ProjectName="Test Project", Description="This is a test project.")
session.add(new_project)

# Insert Sentence
new_sentence = Sentence(Username="john_doe", Text="This is a sample sentence.", ProjectID=1)
session.add(new_sentence)

# Insert Annotation
new_annotation = Annotation(AnnotationText="Sample Annotation", ProjectID=1)
session.add(new_annotation)

# Insert Embedding Model
embed_model = EmbeddingModel(ModelName="SampleEmbedModel", ModelDescription="A sample embedding model description.")
session.add(embed_model)

# Insert Reduction Model
reduction_model = ReductionModel(ModelName="SampleReductionModel", ModelDescription="A sample reduction model description.")
session.add(reduction_model)

# Insert Segment
new_segment = Segment(SentenceID=1, Text="sample", StartPosition=0, AnnotationID=1)
session.add(new_segment)

# Insert Embedding
new_embedding = Embedding(SegmentID=1, ModelID=1, EmbeddingValues=b"1234567890")  # Using a dummy byte string for embedding values
session.add(new_embedding)

# Insert Position
new_position = Position(EmbeddingID=1, ModelID=1, PositionValues=b"0987654321")  # Using a dummy byte string for position values
session.add(new_position)

# Commit the session to save all changes
session.commit()

# Close the session
session.close()