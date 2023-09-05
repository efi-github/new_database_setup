from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import User, Project, Dataset, Sentence, Annotation, Segment, Embedding, Position, View, CombinedModel  # Replace 'your_models_file' with the actual name of your models file
from db.session import SessionLocal

session = SessionLocal()

# Function to pretty print model attributes
def print_model(model_instance):
    print(f"--- {model_instance.__class__.__name__} ---")
    for key, value in model_instance.__dict__.items():
        if not key.startswith('_'):
            print(f"{key}: {value if len(str(value)) < 350 else str(value)[:50] + '...'}")
    print("")

# Query and print all instances for each type
def print_all_instances():
    for model_class in [User, Project, Dataset, Sentence, Annotation, Segment, Embedding, Position, CombinedModel , View]:
        print(f"======== {model_class.__name__} Instances ========")
        for instance in session.query(model_class).all()[:10]:
            print_model(instance)

        print(f"{'=' * 40}\n")

# Specific queries for Project and User based on their IDs
def print_specific_instances(project_id, username):
    print("======== Specific Instances ========\n")

    project_instance = session.query(Project).filter(Project.ProjectID == project_id).first()
    if project_instance:
        print_model(project_instance)
    else:
        print(f"No Project found with ProjectID: {project_id}")

    user_instance = session.query(User).filter(User.Username == username).first()
    if user_instance:
        print_model(user_instance)
    else:
        print(f"No User found with Username: {username}")

    print(f"{'=' * 40}\n")

if __name__ == '__main__':
    print_all_instances()
    print_specific_instances(1, 'string')