from db.models import User, Project, Sentence, Annotation, Segment, Embedding, Position, EmbeddingModel, ReductionModel
from db.session import SessionLocal
from sqlalchemy import create_engine, and_


def add_data_to_db(username, project_name, json_data):
    """
    Add sentences, annotations, and segments to the database based on the provided JSON data.

    :param username: The username of the user.
    :param project_name: The name of the project.
    :param json_data: The JSON data in the provided format.
    :param db_url: The database connection URL.
    """
    session = SessionLocal()

    # Fetch the user and project based on the provided username and project_name
    user = session.query(User).filter_by(Username=username).first()
    project = session.query(Project).filter(
        and_(Project.Username == username, Project.ProjectName == project_name)).first()

    # Check if the user and project exist
    if not user or not project:
        print("User or Project not found in the database!")
        session.close()
        return

    for item in json_data['data']:
        # Add the sentence
        sentence = Sentence(
            Username=user.Username,
            Text=item['text'],
            ProjectID=project.ProjectID
        )
        session.add(sentence)
        session.flush()  # flush to get the generated primary key for sentence

        for entity in item['entities']:
            # Add the annotation
            text = item['text'][entity['start']:entity['end']]
            label = entity['label']
            annotation = Annotation(
                AnnotationText=label,
                ProjectID=project.ProjectID
            )
            session.add(annotation)
            session.flush()  # flush to get the generated primary key for annotation

            # Add the segment
            segment = Segment(
                SentenceID=sentence.SentenceID,
                Text=text,
                StartPosition=entity['start'],
                AnnotationID=annotation.AnnotationID
            )
            session.add(segment)

    session.commit()
    session.close()

json_data = {
    "data": [
        {
            "text": "Barack Obama was born in Hawaii.",
            "entities": [
                {
                    "start": 0,
                    "end": 12,
                    "label": "person"
                },
                {
                    "start": 24,
                    "end": 30,
                    "label": "location"
                }
            ]
        },
        {
            "text": "Apple Inc. was founded in Cupertino.",
            "entities": [
                {
                    "start": 0,
                    "end": 9,
                    "label": "organization"
                },
                {
                    "start": 27,
                    "end": 37,
                    "label": "location"
                }
            ]
        },
        {
            "text": "Mount Everest is the highest peak in the world.",
            "entities": [
                {
                    "start": 0,
                    "end": 12,
                    "label": "location"
                }
            ]
        },
        {
            "text": "The Mona Lisa is displayed at the Louvre.",
            "entities": [
                {
                    "start": 4,
                    "end": 13,
                    "label": "art"
                },
                {
                    "start": 33,
                    "end": 40,
                    "label": "location"
                }
            ]
        }
    ]
}


user = "john_doe"
project = "Test Project"
add_data_to_db(user, project, json_data)
