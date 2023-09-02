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

    return results


input_text = """O	i
O	need
O	that
O	movie
O	which
O	involves
B-Plot	aliens
I-Plot	invading
I-Plot	earth
I-Plot	in
I-Plot	a
I-Plot	particular
I-Plot	united
I-Plot	states
I-Plot	place
I-Plot	in
I-Plot	california

O	what
B-Genre	soviet
I-Genre	science
I-Genre	fiction
B-Opinion	classic
O	about
B-Plot	a
I-Plot	mysterious
I-Plot	planet
O	was
O	later
B-Relationship	remade
O	by
B-Director	steven
I-Director	soderbergh
O	and
B-Actor	george
I-Actor	clooney

O	this
B-Genre	american
I-Genre	classic
O	based
O	on
B-Origin	margaret
I-Origin	mitchell
I-Origin	s
I-Origin	novel
O	had
O	more
O	than
O	50
O	speaking
O	roles
O	and
O	2
O	400
O	extras
O	in
O	the
O	film

O	what
O	is
O	the
O	movie
O	starring
B-Actor	jessica
I-Actor	biel
O	from
B-Year	2003
O	where
O	a
B-Plot	group
I-Plot	of
I-Plot	friends
I-Plot	are
I-Plot	stalked
I-Plot	and
I-Plot	hunted
I-Plot	by
I-Plot	a
I-Plot	deformed
I-Plot	killer
I-Plot	with
I-Plot	a
I-Plot	chainsaw

O	in
O	this
O	movie
B-Actor	ryan
I-Actor	reynolds
O	plays
O	a
B-Plot	superhero
I-Plot	who
I-Plot	must
I-Plot	defend
I-Plot	mankind
I-Plot	from
I-Plot	a
O	a
B-Plot	super
I-Plot	powerful
I-Plot	being
I-Plot	who
I-Plot	feeds
I-Plot	on
I-Plot	fear"""
output = text_to_json(input_text)

import json

print(json.dumps(output, indent=4))

for sentence in output:
    print(sentence['text'])
    for entity in sentence['entities']:
        print(sentence['text'][entity['start']:entity['end']])
        print(entity['label'])
