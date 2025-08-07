import stanza
import os
import logging
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('stanza')
logger.setLevel(logging.DEBUG)

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_reorder')
STANZA_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')
OUTPUT_FILE = os.path.join(CURRENT_DIR, 'reorder_output.txt')

# Initialize mT5 model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

def get_stanza_annotations(words, pretokenized=True):
    """Run Stanza pipeline to get tokens, POS tags, and dependencies with pretokenized input."""
    try:
        # Define paths to custom models and pretrain file
        tokenize_model_path = os.path.join(STANZA_MODEL_DIR, 'si', 'tokenize', 'si_custom_train.pt')
        pos_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
        lemma_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_lemmatizer.pt')
        depparse_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_parser.pt')
        pretrain_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_pretrain.pt')

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
            (lemma_model_path, "Lemmatizer model"),
            (pretrain_path, "Pretrain file")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")

        # Initialize Stanza pipeline
        nlp = stanza.Pipeline(
            lang='si',
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=pretokenized,
            dir=STANZA_MODEL_DIR,
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
            logging_level='DEBUG',
            use_gpu=torch.cuda.is_available()
        )

        # Prepare input for Stanza
        if pretokenized:
            input_data = [words]  # List of lists for single sentence
        else:
            input_data = ' '.join(words)

        # Process the input
        doc = nlp(input_data)
        logging.debug(f"Processed document: {doc}")

        # Extract annotations
        tokens = []
        pos_tags = []
        xpos_tags = []
        deps = []
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                pos_tags.append(word.upos)
                xpos_tags.append(word.xpos if word.xpos else '_')
                deps.append(word.deprel if word.deprel else '_')
        return tokens, pos_tags, xpos_tags, deps

    except Exception as e:
        logging.error(f"Stanza annotation failed: {str(e)}")
        import traceback
        logging.error(f"Stack trace: {traceback.format_exc()}")
        return None, None, None, None

def reorder_sentence(sentence):
    """Use mT5 model to reorder the sentence."""
    # Split sentence into words for pretokenized input
    words = sentence.split()
    
    # Get Stanza annotations
    tokens, pos_tags, xpos_tags, deps = get_stanza_annotations(words, pretokenized=True)
    
    if tokens is None:
        logging.error("Failed to get Stanza annotations, cannot proceed with reordering.")
        return None, None, None, None, None
    
    # Prepare input string
    input_text = f"reorder: {' '.join(tokens)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(deps)}"
    
    # Encode input
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50).to(device)
    
    # Generate output
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    
    # Decode output
    reordered_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tokens, pos_tags, xpos_tags, deps, reordered_sentence

def save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, corrected_sentence):
    """Save the output to a text file in a readable format, appending to existing content."""
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write(f"Input Sentence: {input_sentence}\n")
            f.write(f"Tokens: {tokens}\n")
            f.write(f"POS Tags: {pos_tags}\n")
            f.write(f"XPOS Tags: {xpos_tags}\n")
            f.write(f"Dependencies: {deps}\n")
            f.write(f"Reordered Sentence: {reordered}\n")
            f.write(f"Gender-Corrected Sentence: {corrected_sentence}\n")
            f.write(f"{'='*50}\n\n")
        logging.debug(f"Output successfully appended to {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Failed to save output to {OUTPUT_FILE}: {str(e)}")
        import traceback
        logging.error(f"Stack trace: {traceback.format_exc()}")

def main():
    # Example input sentence (from Brahmi translation)
    input_sentence = "ප්‍රධානි සුමනගුත්ත ලෙණ සංඝයා"
    
    # Get reordered sentence
    tokens, pos_tags, xpos_tags, deps, reordered = reorder_sentence(input_sentence)
    
    if tokens is None:
        print("Failed to process the input sentence.")
        return
    
    # Print results
    print(f"Input Sentence: {input_sentence}")
    print(f"Tokens: {tokens}")
    print(f"POS Tags: {pos_tags}")
    print(f"XPOS Tags: {xpos_tags}")
    print(f"Dependencies: {deps}")
    print(f"Reordered Sentence: {reordered}")

    # Optional: Rule-based gender correction
    def post_process_gender(sentence, xpos_tags):
        output = sentence.split()
        for i, (word, xpos) in enumerate(zip(output, xpos_tags)):
            if xpos == 'TITLE-FEM':
                if word.endswith('ග'):
                    output[i] = word + 'ා'
                elif word.endswith('ල්'):
                    output[i] = word[:-1] + 'ලී'
        return ' '.join(output)
    
    corrected_sentence = post_process_gender(reordered, xpos_tags)
    print(f"Gender-Corrected Sentence: {corrected_sentence}")

    # Save output to file
    save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, corrected_sentence)

if __name__ == "__main__":
    main()