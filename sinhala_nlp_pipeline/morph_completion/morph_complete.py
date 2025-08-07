# from transformers import MT5ForConditionalGeneration, MT5Tokenizer
# import os

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_sinhala')

# def load_mt5():
#     tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
#     model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
#     return model, tokenizer

# def morph_complete(reordered_sentence):
#     model, tokenizer = load_mt5()
#     input_text = f"input: {' '.join(reordered_sentence)} → output:"
#     inputs = tokenizer(input_text, return_tensors='pt')
#     outputs = model.generate(inputs['input_ids'], max_length=50)
#     completed = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return completed.split()

# if __name__ == "__main__":
#     reordered = ['තිස්සා', 'ලෙන', 'සන්ඝයා', 'පුජ']
#     completed = morph_complete(reordered)
#     print(f"Reordered: {reordered}")
#     print(f"Morphologically completed: {completed}")

# from transformers import MT5ForConditionalGeneration, MT5Tokenizer
# import os
# import torch

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_sinhala')

# def load_mt5():
#     tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
#     model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     return model, tokenizer, device

# def morph_complete(reordered_sentence):
#     model, tokenizer, device = load_mt5()
#     input_text = f"input: {' '.join(reordered_sentence)} → output:"
#     inputs = tokenizer(input_text, return_tensors='pt', max_length=50, truncation=True).to(device)
#     outputs = model.generate(
#         inputs['input_ids'],
#         max_length=50,
#         num_beams=5,
#         temperature=0.7,
#         top_k=50,
#         do_sample=True
#     )
#     completed = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     # completed = completed.replace('<extra_id_0>', '').strip()
#     return completed.split()

# if __name__ == "__main__":
#     reordered = ['සුමනගුත්ත', 'රජ', 'කුමරු', 'දත්තා', 'ලෙණ', 'සංඝයා']            
#     # reordered = ['සුමන ගුප්ත', 'සන්ඝයා', 'පූජා', 'ලෙන']
#     completed = morph_complete(reordered)
#     print(f"Reordered: {reordered}")
#     print(f"Morphologically completed: {completed}")

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'mt5_sinhala')
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'morph_results.txt')


def load_mt5():
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device


def morph_complete(sentence):
    model, tokenizer, device = load_mt5()
    input_text = f"input: {sentence} → output:"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=50, truncation=True).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_length=50,
        num_beams=5,
        temperature=0.7,
        top_k=50,
        do_sample=True
    )
    completed = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return completed

def save_to_file(input_sentence, output_sentence):
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"input: {input_sentence} → output: {output_sentence}\n")

if __name__ == "__main__":
    input_sentence = "තිස්ස ප්‍රධානි පුත් දත්ත ලෙණ සංඝයා"
    completed_sentence = morph_complete(input_sentence)
    save_to_file(input_sentence, completed_sentence)
    print(f"Input: {input_sentence}")
    print(f"Output: {completed_sentence}")