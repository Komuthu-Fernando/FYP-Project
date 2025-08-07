from transformers import ByT5Tokenizer, T5ForConditionalGeneration
import os
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala', 'old')
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models', 'byt5_sinhala', 'byt_base_old'))
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'morph_results.txt')

def load_byt5():
    
    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; using default torch loading (ensure PyTorch >= 2.6)")
    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    return model, tokenizer, device

# def clean_output(text):
#     """Fix residual issues in generated output."""
#     text = text.replace('ප් ර', 'ප්‍ර')
#     text = text.replace('න ම්', 'නම්')
#     text = text.replace('පෙ ළ', 'පෙළ')
#     text = text.replace('ගේ ', 'ගේ')
#     text = text.replace('දත්ත ගේ', 'দත්තගේ')
#     return text

def morph_complete(sentence):
    model, tokenizer, device = load_byt5()
    input_text = f"input: {sentence} → output:"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True).to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,  # Allow sampling to generate more complete text
        early_stopping=True
    )
    completed = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # completed = clean_output(completed)
    return completed


def save_to_file(input_sentence, output_sentence):
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"input: {input_sentence} → output: {output_sentence}\n")

if __name__ == "__main__":
    input_sentence = "ප්‍රමුඛ උත්තිය පුත් ප්‍රමුඛ සුම්ම ලෙණ සංඝයා"
    completed_sentence = morph_complete(input_sentence)
    save_to_file(input_sentence, completed_sentence)
    print(f"Input: {input_sentence}")
    print(f"Output: {completed_sentence}")

