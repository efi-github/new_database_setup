from db.models import Segment, Annotation, Sentence, Project, User
from sqlalchemy import and_

def text_to_json(input_text):
    # Split text by double newlines to separate sentences
    sentences = input_text.strip().split('\n\n')

    results = []

    for sentence in sentences:
        lines = sentence.split('\n')

        # Extract the words and labels from each line
        words = [line.split('\t')[1] for line in lines]
        labels = [line.split('\t')[0] for line in lines]

        # Construct the sentence
        text = ' '.join(words)

        entities = []
        start_pos = 0
        in_entity = False

        for i, (word, label) in enumerate(zip(words, labels)):
            if label.startswith('B-'):
                # If we are already in an entity, we close it first
                if in_entity:
                    entities.append({"start": start, "end": start_pos - 1, "label": entity_label})

                # Start of a new entity
                start = start_pos
                entity_label = label[2:]
                in_entity = True
            elif label.startswith('I-') and not in_entity:
                # Continuation of an entity but we missed the beginning
                start = start_pos
                entity_label = label[2:]
                in_entity = True
            elif label.startswith('O') and in_entity:
                # End of the current entity
                entities.append({"start": start, "end": start_pos - 1, "label": entity_label})
                in_entity = False

            start_pos += len(word) + 1  # +1 for the space

        # If the last word was part of an entity
        if in_entity:
            entities.append({"start": start, "end": start_pos - 1, "label": entity_label})

        results.append({"text": text, "entities": entities})

    return {"data": results}

from sqlalchemy import and_

def add_data_to_db(username, project_id, database_name, json_data, session):
    """
    Add sentences, annotations, and segments to the database based on the provided JSON data.

    :param username: The username of the user.
    :param project_id: The ID of the project.
    :param database_name: The name of the dataset.
    :param json_data: The JSON data in the provided format.
    :param session: The SQLAlchemy session object.
    """

    # Fetch the user and project based on the provided username and project_name
    user = session.query(User).filter_by(Username=username).first()
    project = session.query(Project).filter(
        and_(Project.Username == username, Project.ProjectID == project_id)).first()

    # Check if the user and project exist
    if not user or not project:
        print("User or Project not found in the database!")
        session.close()
        return

    # Create a new dataset within the project
    dataset = Dataset(
        ProjectID=project.ProjectID,
        DatasetName=database_name,
    )
    session.add(dataset)
    session.flush()  # flush to get the generated primary key for dataset

    for item in json_data['data']:
        # Add the sentence
        sentence = Sentence(
            Username=user.Username,
            Text=item['text'],
            DatasetID=dataset.DatasetID  # Assign to the new dataset
        )
        session.add(sentence)
        session.flush()  # flush to get the generated primary key for sentence

        for entity in item['entities']:
            # Add the annotation
            text = item['text'][entity['start']:entity['end']]
            label = entity['label']
            annotation_temp = session.query(Annotation).filter_by(AnnotationText=label).first()
            if not annotation_temp:
                annotation_temp = Annotation(
                    AnnotationText=label,
                    ProjectID=project.ProjectID
                )
                session.add(annotation_temp)
                session.flush()  # flush to get the generated primary key for annotation

            # Add the segment
            segment = Segment(
                SentenceID=sentence.SentenceID,
                Text=text,
                StartPosition=entity['start'],
                AnnotationID=annotation_temp.AnnotationID
            )
            session.add(segment)

    session.commit()
    session.close()
