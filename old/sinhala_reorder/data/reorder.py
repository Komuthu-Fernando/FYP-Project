from rules import basic_sov_reorder
import stanza
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def hybrid_reorder(sentence, tagged_sentence, nlp):
    """
    Hybrid Sinhala sentence reordering using rules and dependency parsing.
    """
    # Apply rule-based reordering
    reordered = basic_sov_reorder(tagged_sentence)
    
    if nlp:
        try:
            # Parse the original sentence for dependencies
            doc = nlp(' '.join(sentence))
            
            # Extract dependencies
            words = {word.id: word.text for sent in doc.sentences for word in sent.words}
            dependencies = {word.id: word.head for sent in doc.sentences for word in sent.words}
            
            # Identify root verb(s)
            root_verbs = [words[word_id] for word_id in dependencies if dependencies[word_id] == 0 and word_id in words]
            
            # Ensure verbs are last
            if root_verbs:
                reordered = [word for word in reordered if word not in root_verbs] + root_verbs
        except Exception as e:
            logging.warning(f"Dependency parsing failed: {e}. Falling back to rule-based order.")
    
    return reordered

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            'sentence': ['තිස්සා', 'පුජ', 'ලෙන', 'සන්ඝයා'],
            'tagged': [('තිස්සා', 'PROP'), ('පුජ', 'V'), ('ලෙන', 'PLACE'), ('සන්ඝයා', 'N')]
        },
        {
            'sentence': ['කන්තා', 'ගිහි', 'බැතිමතා', 'තිස්ස', 'ලෙන'],
            'tagged': [('කන්තා', 'N'), ('ගිහි', 'N'), ('බැතිමතා', 'N'), ('තිස්ස', 'PROP'), ('ලෙන', 'PLACE')]
        },
        {
            'sentence': ['සුමන', 'ගුප්ත', 'ලෙන', 'සන්ගයා'],
            'tagged': [('සුමන', 'PROP'), ('ගුප්ත', 'N'), ('ලෙන', 'PLACE'), ('සන්ගයා', 'N')]
        },
        {
            'sentence': ['ගෘහපති', 'වේලු', 'පුතුන්', 'තිදෙනා', 'සහෝදරය', 'සංඝයාට', 'පූජා', 'ලදි', 'ගුහාව'],
            'tagged': [('ගෘහපති', 'TITLE'), ('වේලු', 'PROP'), ('පුතුන්', 'N'), ('තිදෙනා', 'N'), ('සහෝදරය', 'N'), ('සංඝයාට', 'N-DAT'), ('පූජා', 'V'), ('ලදි', 'V'), ('ගුහාව', 'N')]
        }
    ]
    
    # Load Stanza Sinhala model
    nlp = None
    try:
        nlp = stanza.Pipeline('si', processors='tokenize,pos,depparse', package='default', tokenize_no_mwt=True)
    except Exception as e:
        logging.error(f"Failed to load Stanza pipeline: {e}. Using rule-based reordering only.")
    
    # Reorder each test case
    for test in test_cases:
        sentence = test['sentence']
        tagged_sentence = test['tagged']
        reordered = hybrid_reorder(sentence, tagged_sentence, nlp)
        print(f"Original: {sentence}")
        print(f"Reordered: {reordered}\n")