import os
import logging
import unicodedata
import re
import torch
import pandas as pd
import stanza
from transformers import ByT5Tokenizer, T5ForConditionalGeneration

# === Logging ===
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('stanza')
logger.setLevel(logging.DEBUG)

# === Paths ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REORDER_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_reorder')
MORPH_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala', 'byt_base_old')
# MORPH_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala', 'final', 'epoch_2')
STANZA_MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')
OUTPUT_FILE = os.path.join(CURRENT_DIR, 'final_output.txt')
DICTIONARY_FILE = os.path.join(CURRENT_DIR, 'dictionary.xlsx')

def clean_unicode_text(text):
    # Normalize to NFC (or NFKC for compatibility forms)
    text = unicodedata.normalize('NFC', text)
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Dictionary Mapping ===
def map_words_to_meanings(word_list):
    try:
        df = pd.read_excel(DICTIONARY_FILE)
        word_map = dict(zip(df['Brhami Word'], df['Meaning']))
        meanings = []
        missing_words = []

        for word in word_list:
            if word in word_map:
                meanings.append(word_map[word])
            else:
                missing_words.append(word)

        if missing_words:
            error_message = f"Error: The following words were not found in the Excel file: {', '.join(missing_words)}"
            with open('error_log.txt', 'w') as f:
                f.write(error_message)
            return None

        # return clean_unicode_text(' '.join(meanings))
        return ' '.join(meanings)
    except Exception as e:
        logging.error(f"Dictionary mapping failed: {str(e)}")
        return None

# === Load ByT5 Models ===
def load_model(model_dir):
    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; fallback to default")

    model_dir = os.path.abspath(model_dir)
    tokenizer = ByT5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    return model, tokenizer, device

# Load models
reorder_model, reorder_tokenizer, reorder_device = load_model(REORDER_MODEL_DIR)
morph_model, morph_tokenizer, morph_device = load_model(MORPH_MODEL_DIR)

# === Morphological Completion ===
def morph_complete(sentence):
    input_text = f"input: {sentence} → output:"
    inputs = morph_tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(morph_device)
    outputs = morph_model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        early_stopping=True
    )
    return morph_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# === Stanza Processing ===
def get_stanza_annotations(words, pretokenized=True):
    try:
        tokenize_model_path = os.path.join(STANZA_MODEL_DIR, 'si', 'tokenize', 'si_custom_train.pt')
        pos_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_tagger.pt')
        lemma_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_lemmatizer.pt')
        depparse_model_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_nocharlm_parser.pt')
        pretrain_path = os.path.join(STANZA_MODEL_DIR, 'si_custom_pretrain.pt')

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
        return None, None, None, None

# === Sentence Reordering ===
def reorder_sentence(sentence):
    words = sentence.split()
    tokens, pos_tags, xpos_tags, deps = get_stanza_annotations(words, pretokenized=True)

    if tokens is None:
        logging.error("Stanza failed. Cannot reorder.")
        return None, None, None, None, None

    input_text = f"{' '.join(tokens)} | POS: {' '.join(pos_tags)} | XPOS: {' '.join(xpos_tags)} | Dep: {' '.join(deps)}"
    full_input = f"input: {input_text} → output:"
    inputs = reorder_tokenizer(full_input, return_tensors='pt', padding=True, truncation=True, max_length=256).to(reorder_device)

    reorder_model.eval()
    with torch.no_grad():
        outputs = reorder_model.generate(
            input_ids=inputs['input_ids'],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    reordered = reorder_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tokens, pos_tags, xpos_tags, deps, reordered

# === Output Logging ===
def save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gender_corrected, morph_completed):
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write(f"Input Sentence: {input_sentence}\n")
            # f.write(f"Tokens: {tokens}\n")
            f.write(f"POS Tags: {pos_tags}\n")
            f.write(f"XPOS Tags: {xpos_tags}\n")
            f.write(f"Dependencies: {deps}\n")
            f.write(f"Reordered Sentence: {reordered}\n")
            f.write(f"Gender-Corrected Sentence: {gender_corrected}\n")
            f.write(f"Morphologically Completed Sentence: {morph_completed}\n")
            f.write(f"{'='*50}\n\n")
    except Exception as e:
        logging.error(f"Failed to save output: {str(e)}")

# === Main ===
def main():
    brhami_words = ['පරුමක', 'තිශ', 'ලෙණෙ', 'ශගශ']  # Replace with your list
    input_sentence = map_words_to_meanings(brhami_words)
    # input_sentence = clean_unicode_text(input_sentence)

    if not input_sentence:
        print("Word-to-meaning mapping failed. Check error_log.txt.")
        return

    tokens, pos_tags, xpos_tags, deps, reordered = reorder_sentence(input_sentence)
    if tokens is None:
        print("Reordering failed.")
        return

    print(f"Input Sentence: {input_sentence}")
    print(f"POS Tags: {pos_tags}")
    print(f"XPOS Tags: {xpos_tags}")
    print(f"Dependencies: {deps}")
    print(f"Reordered Sentence: {reordered}")

    # === Gender Correction ===
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

    # === Morph Completion ===
    morph_completed = morph_complete(gender_corrected)
    print(f"Morphologically Completed Sentence: {morph_completed}")

    save_output(input_sentence, tokens, pos_tags, xpos_tags, deps, reordered, gender_corrected, morph_completed)

if __name__ == "__main__":
    main()
