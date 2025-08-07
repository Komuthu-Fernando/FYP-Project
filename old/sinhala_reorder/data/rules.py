def basic_sov_reorder(tagged_sentence):
    """
    Reorder a tagged Sinhala sentence into SOV order with improved logic.
    """
    # Subjects: Collect all PROP and TITLE tags as subject phrase
    subjects = [word for word, tag in tagged_sentence if tag in ['PROP', 'TITLE']]
    
    # Verbs: All V tags, placed last
    verbs = [word for word, tag in tagged_sentence if tag == 'V']
    
    # Objects: Indirect (N-DAT) after Direct (N, PLACE)
    direct_objects = [word for word, tag in tagged_sentence if tag in ['N', 'PLACE'] and word not in subjects]
    indirect_objects = [word for word, tag in tagged_sentence if tag == 'N-DAT' and word not in subjects]
    
    # Modifiers: Remaining N as descriptors (e.g., kinship terms), ADJ, CONJ
    descriptors = [word for word, tag in tagged_sentence if tag == 'N' and word not in subjects + direct_objects + indirect_objects]
    adjectives = [word for word, tag in tagged_sentence if tag == 'ADJ']
    conjunctions = [word for word, tag in tagged_sentence if tag == 'CONJ']
    
    # Order: Subjects + Descriptors + Adjectives + Direct Objects + Indirect Objects + Conjunctions + Verbs
    return subjects + descriptors + adjectives + direct_objects + indirect_objects + conjunctions + verbs