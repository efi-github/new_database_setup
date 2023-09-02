from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal


def print_all_users(db_url):
    """
    Print all segments from the database.

    :param db_url: The database connection URL
    """
    # Create a session to interact with the DB

    session = SessionLocal()

    # Query all segments
    users = session.query(User).all()

    for user in users:
        print(f"Username: {user.Username}")
        print(f"Email: {user.Email}")
        print(f"Password: {user.Password}")
        print("-------------------------------")

    session.close()


# Example usage:
DB_URL = "your_database_url_here"
print_all_users(DB_URL)
