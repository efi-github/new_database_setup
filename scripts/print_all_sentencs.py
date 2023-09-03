from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal


def print_all_sentences():
    """
    Print all segments from the database.

    :param db_url: The database connection URL
    """
    # Create a session to interact with the DB

    session = SessionLocal()

    # Query all segments
    segments = session.query(Sentence).all()

    for sentence in segments:
        print(f"Sentence ID: {sentence.SentenceID}")
        print(f"Text: {sentence.Text}")
        print(f"Dataset ID: {sentence.DatasetID}")
        print(f"Position in Dataset: {sentence.PositionInProject}")
        print(f"Username: {sentence.Username}")
        print(sentence.segments)
        for segment in sentence.segments:
            print(f"Segment ID: {segment.SegmentID}")
            print(f"Text: {segment.Text}")
            print(f"Start Position: {segment.StartPosition}")
            print(f"Annotation ID: {segment.AnnotationID}")
            annotation = segment.annotation
            if annotation:
                print(f"Annotation Text: {annotation.AnnotationText}")
                print(f"Project ID: {annotation.ProjectID}")
            print("-------------------------------")

        print("-------------------------------")

    session.close()



print_all_sentences()
