import stanza
import os
import logging
import torch

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('stanza')
logger.setLevel(logging.DEBUG)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))

def parse_dependencies(words, pretokenized=True):
    """
    Parse dependencies using Stanza with pretokenized input and custom models.
    
    Args:
        words: List of words (pretokenized)
        pretokenized: Boolean to indicate if input is pretokenized (default: True)
    
    Returns:
        List of dependency dictionaries or None if parsing fails
    """
    try:
        # Define paths to custom models and pretrain file
        tokenize_model_path = os.path.join(MODEL_DIR, 'si/tokenize/si_custom_train.pt')
        pos_model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
        depparse_model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_parser.pt')
        lemma_model_path = os.path.join(MODEL_DIR, 'si_custom_nocharlm_lemmatizer.pt')
        pretrain_path = os.path.join(MODEL_DIR, 'si_custom_pretrain.pt')

        # Log paths and verify they exist
        logging.debug(f"Tokenize model path: {tokenize_model_path}, Exists: {os.path.exists(tokenize_model_path)}")
        logging.debug(f"POS model path: {pos_model_path}, Exists: {os.path.exists(pos_model_path)}")
        logging.debug(f"Depparse model path: {depparse_model_path}, Exists: {os.path.exists(depparse_model_path)}")
        logging.debug(f"Lemma model path: {lemma_model_path}, Exists: {os.path.exists(lemma_model_path)}")
        logging.debug(f"Pretrain path: {pretrain_path}, Exists: {os.path.exists(pretrain_path)}")

        # Check if all required files exist
        for path, name in [
            (tokenize_model_path, "Tokenizer model"),
            (pos_model_path, "POS model"),
            (depparse_model_path, "Depparse model"),
            ((lemma_model_path), "Lemmatizer model"),
            (pretrain_path, "Pretrain file")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")

        # Initialize Stanza pipeline
        nlp = stanza.Pipeline(
            lang='si',
            processors='lemma,tokenize,pos,depparse',
            tokenize_pretokenized=pretokenized,
            dir=MODEL_DIR,
            package='si_custom',
            tokenize_model_path=tokenize_model_path,
            lemma_model_path=lemma_model_path,
            pos_model_path=pos_model_path,
            pos_pretrain_path=pretrain_path,
            depparse_model_path=depparse_model_path,
            depparse_pretrain_path=pretrain_path,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            use_cache=False,
            verbose=True,
            logging_level='DEBUG'
        )

        # Prepare input for Stanza
        if pretokenized:
            input_data = [words]  # List of lists for single sentence
        else:
            input_data = ' '.join(words)

        # Process the input
        doc = nlp(input_data)
        logging.debug(f"Processed document: {doc}")

        # Extract dependencies
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
        logging.error(f"Dependency parsing failed: {str(e)}")
        import traceback
        logging.error(f"Stack trace: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Test with sample input
    words = ['ප්‍රධානියා', 'සිව', 'ලෙණ‌', 'සංඝයා' ]
    deps = parse_dependencies(words)
    print(f"Input words: {words}")
    print(f"Dependencies: {deps}")