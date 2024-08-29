def extract_entities_from_bio(tokens, bio_tags):
    entities = []
    current_entity = None
    current_type = None

    for token, tag in zip(tokens, bio_tags):
        if tag == 'O':
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = None
                current_type = None
        elif tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = token
            current_type = tag.split('-')[1]
        elif tag.startswith('I-'):
            if current_entity and current_type == tag.split('-')[1]:
                current_entity += token
        else:
            raise ValueError("Invalid BIO tag: {}".format(tag))

    if current_entity:
        entities.append((current_entity, current_type))

    return entities


tokens = '我在成都双流区，我要去山东省济南市办理身份证'
tokens = [i for i in tokens]
bio_tags = ['O', 'O', 'B-area', 'I-area', 'I-area', 'I-area', 'I-area', 'O', 'O', 'O', 'O', 'B-pro', 'I-pro', 'I-pro',
            'B-market', 'I-market', 'I-market', 'B-gov', 'I-gov', 'I-gov', 'I-gov', 'I-gov']

entities = extract_entities_from_bio(tokens, bio_tags)
print(entities)
