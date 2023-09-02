import pytest
from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position
from db.session import SessionLocal

@pytest.fixture(scope='module')
def db_session():
    session = SessionLocal()
    yield session
    session.close()

def test_insert_data(db_session):
    # create a user
    user = User(Username="JohnDoe", Email="johndoe@example.com", Password="securepassword")
    db_session.add(user)
    db_session.commit()

    # create a project associated with the user
    project = Project(Username=user.Username, ProjectName="Test Project", Description="This is a test project.")
    db_session.add(project)
    db_session.commit()

    # create a sentence associated with the user and project
    sentence = Sentence(Username=user.Username, Text="This is a test sentence.", ProjectID=project.ProjectID)
    db_session.add(sentence)
    db_session.commit()

    # create an annotation associated with the project
    annotation = Annotation(AnnotationText="Test Annotation", ProjectID=project.ProjectID)
    db_session.add(annotation)
    db_session.commit()

    # create a segment associated with the sentence and annotation
    segment = Segment(SentenceID=sentence.SentenceID, Text="test segment", StartPosition=1, AnnotationID=annotation.AnnotationID)
    db_session.add(segment)
    db_session.commit()

    # create an embedding for the segment
    embedding = Embedding(SegmentID=segment.SegmentID, EmbeddingAlgorithm="Test Algorithm", EmbeddingValues=b"test data")
    db_session.add(embedding)
    db_session.commit()

    # create a position for the embedding
    position = Position(EmbeddingID=embedding.EmbeddingID, DimensionReductionAlgorithm="Test Algorithm", PositionValues=b"test position data")
    db_session.add(position)
    db_session.commit()

def test_cascading_properties(db_session):
    # Fetch the user we previously added
    user = db_session.query(User).filter(User.Username == "JohnDoe").first()

    assert user is not None, "User not found!"

    # Delete the user
    db_session.delete(user)
    db_session.commit()

    # Check if related data has been deleted due to cascading
    project = db_session.query(Project).filter(Project.Username == user.Username).first()
    sentence = db_session.query(Sentence).filter(Sentence.Username == user.Username).first()

    assert project is None and sentence is None, "Cascade delete failed!"
