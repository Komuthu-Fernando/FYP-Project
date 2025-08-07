import stanza
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Download specific processors for Sinhala
    stanza.download('si', processors='tokenize,pos', model_dir='/Users/ajithfernando/stanza_resources', logging_level='INFO')
    
    # Initialize pipeline with tokenize and pos
    nlp = stanza.Pipeline('si', processors='tokenize,pos', model_dir='/Users/ajithfernando/stanza_resources', use_gpu=False)
    
    # Test with a sample Sinhala sentence
    test_sentence = "මම ගෙදර යනවා"  # "I go home"
    doc = nlp(test_sentence)
    
    # Print results
    for sent in doc.sentences:
        print("Words and POS tags:")
        for word in sent.words:
            print(f"ID: {word.id}\tText: {word.text}\tPOS: {word.upos}")
except Exception as e:
    logging.error(f"Error in Stanza pipeline: {str(e)}")
    raise