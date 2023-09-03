from db.models import Segment, Annotation, Sentence, Project, User, Dataset
from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert
import time

def text_to_json(input_text, options=None):
    # Split text by double newlines to separate sentences
    sentences = input_text.strip().split(options.sentence_split)
    results = []

    for sentence in sentences:
        lines = sentence.split('\n')

        # Extract the words and labels from each line
        words = [line.split(options.split)[options.word_idx].strip() for line in lines]
        labels = [line.split(options.split)[options.label_idx].strip() for line in lines]

        # Construct the sentence
        text = ' '.join(words)
        if options.type == "plain":
            entities = []
            start_pos = 0
            in_entity = False
            entity_label = ""
            for i, (word, label) in enumerate(zip(words, labels)):
                if label != "O":
                    # If we are already in an entity, we close it first
                    if not in_entity:
                        start = start_pos
                        entity_label = label
                        in_entity = True
                    elif entity_label != label:
                        entities.append({"start": start, "end": start_pos - 1, "label": entity_label})
                        start = start_pos
                        entity_label = label
                else:
                    if in_entity:
                        entities.append({"start": start, "end": start_pos - 1, "label": entity_label})
                        in_entity = False
                start_pos += len(word) + 1
            if in_entity:
                entities.append({"start": start, "end": start_pos - 1, "label": entity_label})

            results.append({"text": text, "entities": entities})


        if options.type == "B-I-O":
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

    #for res in results:
    #    print(res['text'])
    #    for entities in res['entities']:
    #        print(f"{entities['start']} - {entities['end']} : {entities['label']}")
    #        print(f"'{res['text'][entities['start']:entities['end']]}'")

    return {"data": results}

def add_data_to_db(username, project_id, database_name, json_data, session):
    start_time = time.time()
    user = session.query(User).filter_by(Username=username).first()
    project = session.query(Project).filter(
        and_(Project.Username == username, Project.ProjectID == project_id)).first()
    query_time = time.time()
    print(f"Querying the database took {query_time - start_time} seconds.")


    if not user or not project:
        print("User or Project not found in the database!")
        session.close()
        return

    dataset = Dataset(
        ProjectID=project.ProjectID,
        DatasetName=database_name,
    )
    session.add(dataset)
    session.commit()

    database_time = time.time()
    print(f"Adding the dataset to the database took {database_time - query_time} seconds.")

    sentence_dicts = [
        {
            'Username': user.Username,
            'Text': item['text'],
            'DatasetID': dataset.DatasetID,
            'PositionInProject': i
        }
        for i, item in enumerate(json_data['data'])
    ]

    insert_stmt = insert(Sentence).values(sentence_dicts).returning(Sentence.SentenceID)
    sentence_ids = [row[0] for row in session.execute(insert_stmt).fetchall()]

    sentence_time = time.time()
    print(f"Adding the sentences to the database took {sentence_time - database_time} seconds.")


    segment_dicts = []
    annotations_dict = {a.AnnotationText: a.AnnotationID for a in
                        session.query(Annotation).filter_by(ProjectID=project.ProjectID).all()}
    new_annotations = set()

    for item, sentence_id in zip(json_data['data'], sentence_ids):
        for entity in item['entities']:
            label = entity['label']

            annotation_id = annotations_dict.get(label)

            # If the annotation doesn't exist, add it to the database and the dictionary
            if annotation_id is None:
                if label not in new_annotations:
                    new_annotation = Annotation(
                        AnnotationText=label,
                        ProjectID=project.ProjectID
                    )
                    session.add(new_annotation)
                    session.commit()
                    annotation_id = new_annotation.AnnotationID
                    annotations_dict[label] = annotation_id
                    new_annotations.add(label)

            segment_dict = {
                'SentenceID': sentence_id,
                'Text': item['text'][entity['start']:entity['end']],
                'StartPosition': entity['start'],
                'AnnotationID': annotation_id
            }

            segment_dicts.append(segment_dict)

    if segment_dicts:
        session.bulk_insert_mappings(Segment, segment_dicts)

    session.commit()
    session.close()

    segment_time = time.time()
    print(f"Adding the segments to the database took {segment_time - sentence_time} seconds.")
    print(f"Added {len(json_data['data'])} sentences to the database.")
    print(f"Adding the data to the database took {time.time() - start_time} seconds.")


def add_data_to_db1(username, project_id, database_name, json_data, session):
    start_time = time.time()
    user = session.query(User).filter_by(Username=username).first()
    project = session.query(Project).filter(
        and_(Project.Username == username, Project.ProjectID == project_id)).first()
    query_time = time.time()
    print(f"Querying the database took {query_time - start_time} seconds.")


    if not user or not project:
        print("User or Project not found in the database!")
        session.close()
        return

    dataset = Dataset(
        ProjectID=project.ProjectID,
        DatasetName=database_name,
    )
    session.add(dataset)
    session.commit()

    database_time = time.time()
    print(f"Adding the dataset to the database took {database_time - query_time} seconds.")

    sentence_dicts = [
        {
            'Username': user.Username,
            'Text': item['text'],
            'DatasetID': dataset.DatasetID
        }
        for item in json_data['data']
    ]

    insert_stmt = insert(Sentence).values(sentence_dicts).returning(Sentence.SentenceID)
    sentence_ids = [row[0] for row in session.execute(insert_stmt).fetchall()]

    sentence_time = time.time()
    print(f"Adding the sentences to the database took {sentence_time - database_time} seconds.")

    # Assuming order is maintained
    for item, sentence_id in zip(json_data['data'], sentence_ids):
        for entity in item['entities']:
            label = entity['label']
            annotation = session.query(Annotation).filter_by(AnnotationText=label).first()
            if not annotation:
                annotation = Annotation(
                    AnnotationText=label,
                    ProjectID=project.ProjectID
                )
                session.add(annotation)
                session.commit()

            segment_dict = {
                'SentenceID': sentence_id,
                'Text': item['text'][entity['start']:entity['end']],
                'StartPosition': entity['start'],
                'AnnotationID': annotation.AnnotationID
            }

            session.execute(insert(Segment).values(segment_dict))

    segment_time = time.time()
    print(f"Adding the segments to the database took {segment_time - sentence_time} seconds.")

    session.commit()
    session.close()
    print(f"Added {len(json_data['data'])} sentences to the database.")
    print(f"Adding the data to the database took {time.time() - start_time} seconds.")
