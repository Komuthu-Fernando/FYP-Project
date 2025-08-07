from transformers import ByT5Tokenizer, T5ForConditionalGeneration
import os
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models', 'byt5_reorder'))
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'byt5_reorder_results.txt')

def load_model():
    try:
        from safetensors.torch import load_file
        print("Using safetensors for model loading")
    except ImportError:
        print("safetensors not installed; falling back to default loader")

    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded to device: {device}")
    return model, tokenizer, device

def reorder_sentence(input_text):
    model, tokenizer, device = load_model()

    formatted_input = f"input: {input_text} → output:"
    inputs = tokenizer(formatted_input, return_tensors='pt', truncation=True, max_length=256).to(device)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=256,
        num_beams=4,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        early_stopping=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def save_result(input_text, output_text):
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"input: {input_text} → output: {output_text}\n")

if __name__ == "__main__":
    # Example input used during training format (without POS, XPOS, DEP here)
    input_sentence = "ප්‍රධානි සුමනගුත්ත ලෙණ සංඝයා | POS: NOUN PROPN NOUN NOUN | XPOS: TITLE-MASC N-MASC _ _ | Dep: appos nmod root nmod"
    reordered_output = reorder_sentence(input_sentence)
    save_result(input_sentence, reordered_output)

    print(f"Input: {input_sentence}")
    print(f"Output: {reordered_output}")
