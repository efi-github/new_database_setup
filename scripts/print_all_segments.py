from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal


def print_all_segments(db_url):
    """
    Print all segments from the database.

    :param db_url: The database connection URL
    """
    # Create a session to interact with the DB

    session = SessionLocal()

    # Query all segments
    segments = session.query(Segment).all()

    for segment in segments:
        print(f"Segment ID: {segment.SegmentID}")
        print(f"Text: {segment.Text}")
        print(f"Start Position: {segment.StartPosition}")
        print(f"Sentence ID: {segment.SentenceID}")
        print(f"Annotation ID: {segment.AnnotationID}")
        annotation = segment.annotation
        if annotation:
            print(f"Annotation Text: {annotation.AnnotationText}")
            print(f"Project ID: {annotation.ProjectID}")
        sentence = segment.sentence
        if sentence:
            #print(f"Sentence ID: {sentence.SentenceID}")
            print(f"Text: {sentence.Text}")
        print("-------------------------------")

    session.close()


# Example usage:
DB_URL = "your_database_url_here"
print_all_segments(DB_URL)
