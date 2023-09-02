from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal


def print_all_annotations(db_url):
    """
    Print all annotations from the database.

    :param db_url: The database connection URL
    """
    # Create a session to interact with the DB

    session = SessionLocal()

    # Query all annotations
    annotations = session.query(Annotation).all()

    for an in annotations:
        print(f"Annotation ID: {an.AnnotationID}")
        print(f"Annotation Text: {an.AnnotationText}")

    session.close()


# Example usage:
DB_URL = "your_database_url_here"
print_all_annotations(DB_URL)
