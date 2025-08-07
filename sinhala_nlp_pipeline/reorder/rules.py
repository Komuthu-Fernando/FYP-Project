def basic_sov_reorder(tagged_sentence, dependencies=None):
    subjects = [word for word, tag in tagged_sentence if tag in ['PROP', 'TITLE', 'PRON']]
    verbs = [word for word, tag in tagged_sentence if tag == 'V']
    direct_objects = [word for word, tag in tagged_sentence if tag in ['N', 'PLACE'] and word not in subjects]
    indirect_objects = [word for word, tag in tagged_sentence if tag == 'N-DAT' and word not in subjects]
    descriptors = [word for word, tag in tagged_sentence if tag == 'N' and word not in subjects + direct_objects + indirect_objects]
    adjectives = [word for word, tag in tagged_sentence if tag == 'ADJ']
    conjunctions = [word for word, tag in tagged_sentence if tag == 'CONJ']
    
    if dependencies:
        root_verbs = [d['word'] for d in dependencies if d['head'] == 0 and d['word'] in verbs]
        if root_verbs:
            reordered = [w for w in subjects + descriptors + adjectives + direct_objects + indirect_objects + conjunctions if w not in root_verbs] + root_verbs
            return reordered
    
    return subjects + descriptors + adjectives + direct_objects + indirect_objects + conjunctions + verbs