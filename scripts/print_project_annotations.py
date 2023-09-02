from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal

def print_annotations_for_project(project_id, db_url):
    """
    Print all annotations associated with a specific project from the database.

    :param project_id: The ID of the project for which annotations are to be printed
    :param db_url: The database connection URL
    """
    # Create a session to interact with the DB
    session = SessionLocal()

    # Query the specific project
    project = session.query(Project).filter_by(ProjectID=project_id).first()

    if not project:
        print(f"No project found with ID: {project_id}")
        session.close()
        return

    # Print annotations related to this project
    for annotation in project.annotations:
        print(f"Annotation ID: {annotation.AnnotationID}")
        print(f"Text: {annotation.AnnotationText}")
        print("-------------------------------")

    session.close()


# Example usage:
DB_URL = "your_database_url_here"
PROJECT_ID = 1# Replace with the desired project ID
print_annotations_for_project(PROJECT_ID, DB_URL)