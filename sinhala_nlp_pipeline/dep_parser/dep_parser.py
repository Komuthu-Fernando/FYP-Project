import stanza
import os
import logging

logging.basicConfig(level=logging.DEBUG)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))

def parse_dependencies(tagged_sentence, pretokenized=True):
    """
    Parse dependencies using Stanza with pre-tagged input, using custom models.
    
    Args:
        tagged_sentence: List of (word, tag) tuples from hybrid_tagger
        pretokenized: Boolean to use pretokenized input
    
    Returns:
        List of dependency dictionaries or None if failed
    """
    try:
        model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_parser.pt')
        logging.debug(f"Model path: {model_path}, Exists: {os.path.exists(model_path)}")
        
        # Configure pipeline with all required processors and custom models
        nlp = stanza.Pipeline(
            lang='si',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=pretokenized,
            pos_pretagged=True,  # Use provided POS tags
            lemma_pretagged=True,  # Assume lemma is not critical; set to True for now
            dir=MODEL_DIR,
            package='si_custom',
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            use_cache=False
        )
        
        # Prepare pretokenized input with POS tags (lemma can be placeholder '_')
        pretokenized_input = '\n'.join([f"{word}\t_\t{tag}" for word, tag in tagged_sentence])
        doc = nlp(pretokenized_input)
        
        logging.debug(f"Doc: {doc}")
        
        dependencies = []
        for sent in doc.sentences:
            for word in sent.words:
                dependencies.append({
                    'word': word.text,
                    'id': word.id,
                    'head': word.head,
                    'deprel': word.deprel
                })
        return dependencies
    except Exception as e:
        logging.error(f"Dependency parsing failed: {e}")
        return None

if __name__ == "__main__":
    # Test with sample tagged input
    tagged_sentence = [('තිස්සා', 'PROP'), ('පුජ', 'V'), ('ලෙන', 'PLACE'), ('සන්ඝයා', 'N')]
    deps = parse_dependencies(tagged_sentence)
    print(f"Tagged: {tagged_sentence}")
    print(f"Dependencies: {deps}")