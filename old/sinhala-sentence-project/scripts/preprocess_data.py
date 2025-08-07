# import os
# from sinling import SinhalaTokenizer

# tokenizer = SinhalaTokenizer()

# def preprocess_mt5_data(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
#         for line in f_in:
#             input_text, target = line.strip().split('\t')
#             # Ensure input is sorted for consistency
#             words = input_text.replace('words: ', '').split()
#             sorted_words = ' '.join(sorted(words))
#             f_out.write(f"{input_text}\t{target}\n")

# # Process mT5 training and test data
# preprocess_mt5_data('data/raw/train_pairs.txt', 'data/processed/mt5_train_data.txt')
# preprocess_mt5_data('data/raw/test_pairs.txt', 'data/processed/mt5_test_data.txt')

import os
from sinling import SinhalaTokenizer

tokenizer = SinhalaTokenizer()

def preprocess_mt5_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in, 1):  # Track line number
            # Skip empty lines
            if not line.strip():
                print(f"Warning: Skipping empty line {i} in {input_file}")
                continue
            # Split line, handle potential errors
            try:
                input_text, target = line.strip().split('\t')
                if not input_text.startswith('words: '):
                    print(f"Warning: Line {i} in {input_file} missing 'words: ' prefix: {line.strip()}")
                    continue
                if not target:
                    print(f"Warning: Line {i} in {input_file} missing target sentence: {line.strip()}")
                    continue
                # Ensure input is sorted for consistency
                words = input_text.replace('words: ', '').split()
                sorted_words = ' '.join(sorted(words))
                f_out.write(f"words: {sorted_words}\t{target}\n")
            except ValueError:
                print(f"Error: Line {i} in {input_file} has invalid format (needs tab-separated input and output): {line.strip()}")
                continue

# Process mT5 training and test data
preprocess_mt5_data('data/raw/train_pairs.txt', 'data/processed/mt5_train_data.txt')
preprocess_mt5_data('data/raw/test_pairs.txt', 'data/processed/mt5_test_data.txt')