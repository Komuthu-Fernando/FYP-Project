import stanza
import os
import logging
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, ByT5Tokenizer, T5ForConditionalGeneration

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('stanza')
logger.setLevel(logging.DEBUG)

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MT5_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_reorder')
BYT5_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala', 'byt_base_old')
STANZA_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')
OUTPUT_FILE = os.path.join(CURRENT_DIR, 'reorder_output.txt')

# Initialize mT5 model and tokenizer
mt5_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mt5_tokenizer = MT5Tokenizer.from_pretrained(MT5_MODEL_DIR)
mt5_model = MT5ForConditionalGeneration.from_pretrained(MT5_MODEL_DIR).to(mt5_device)

# Load ByT5 model and tokenizer
def load_byt5():
    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; using default torch loading")

    # Normalize the path to avoid HuggingFace misinterpreting it
    model_dir = os.path.abspath(BYT5_MODEL_DIR)

    tokenizer = ByT5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    return model, tokenizer, device

byt5_model, byt5_tokenizer, byt5_device = load_byt5()

def morph_complete(sentence):
    input_text = f"input: {sentence} → output:"
    inputs = byt5_tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(byt5_device)
    outputs = byt5_model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        early_stopping=True
    )
    completed = byt5_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return completed

def get_stanza_annotations(words, pretokenized=True):
    try:
        tokenize_model_path = os.path.join(STANZA_MODEL_DIR, 'si', 'tokenize', 'si_custom_train.pt')
        pos_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
        lemma_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_lemmatizer.pt')
        depparse_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_parser.pt')
        pretrain_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_pretrain.pt')

        for path, name in [
            (tokenize_model_path, "Tokenizer model"),
            (pos_model_path, "POS model"),
            (depparse_model_path, "Depparse model"),
            (lemma_model_path, "Lemmatizer model"),
            (pretrain_path, "Pretrain file")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")

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

        input_data = [words] if pretokenized else ' '.join(words)
        doc = nlp(input_data)

        tokens, pos_tags, xpos_tags, deps = [], [], [], []
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
    words = sentence.split()
    tokens, pos_tags, xpos_tags, deps = get_stanza_annotations(words, pretokenized=True)

    if tokens is None:
        logging.error("Failed to get Stanza annotations, cannot proceed with reordering.")
        return None, None, None, None, None

    input_text = f"reorder: {' '.join(tokens)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(deps)}"
    inputs = mt5_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50).to(mt5_device)

    mt5_model.eval()
    with torch.no_grad():
        outputs = mt5_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

    reordered_sentence = mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tokens, pos_tags, xpos_tags, deps, reordered_sentence

def save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gender_corrected, morph_completed):
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write(f"Input Sentence: {input_sentence}\n")
            f.write(f"Tokens: {tokens}\n")
            f.write(f"POS Tags: {pos_tags}\n")
            f.write(f"XPOS Tags: {xpos_tags}\n")
            f.write(f"Dependencies: {deps}\n")
            f.write(f"Reordered Sentence: {reordered}\n")
            f.write(f"Gender-Corrected Sentence: {gender_corrected}\n")
            f.write(f"Morphologically Completed Sentence: {morph_completed}\n")
            f.write(f"{'='*50}\n\n")
        logging.debug(f"Output successfully appended to {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Failed to save output: {str(e)}")
        import traceback
        logging.error(f"Stack trace: {traceback.format_exc()}")

def main():
    input_sentence = "වෙළෙන්දා සුමන ලෙණ"
    tokens, pos_tags, xpos_tags, deps, reordered = reorder_sentence(input_sentence)

    if tokens is None:
        print("Failed to process the input sentence.")
        return

    print(f"Input Sentence: {input_sentence}")
    print(f"Tokens: {tokens}")
    print(f"POS Tags: {pos_tags}")
    print(f"XPOS Tags: {xpos_tags}")
    print(f"Dependencies: {deps}")
    print(f"Reordered Sentence: {reordered}")

    def post_process_gender(sentence, xpos_tags):
        output = sentence.split()
        for i, (word, xpos) in enumerate(zip(output, xpos_tags)):
            if xpos == 'TITLE-FEM':
                if word.endswith('ග'):
                    output[i] = word + 'ා'
                elif word.endswith('ල්'):
                    output[i] = word[:-1] + 'ලී'
        return ' '.join(output)

    gender_corrected = post_process_gender(reordered, xpos_tags)
    print(f"Gender-Corrected Sentence: {gender_corrected}")

    morph_completed = morph_complete(gender_corrected)
    print(f"Morphologically Completed Sentence: {morph_completed}")

    save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gender_corrected, morph_completed)

if __name__ == "__main__":
    main()
