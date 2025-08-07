import csv
import re

# Input and output file paths
input_file = "filtered_inscriptions.csv"
output_file = "word_meaning_mapping.csv"

# Function to clean and split transliteration into words
def split_transliteration(text):
    # Remove numbers in parentheses and square brackets
    text = re.sub(r'\(\d+\)', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    # Split into words
    words = [word.strip() for word in text.split() if word.strip()]
    return words

# Function to clean and split translation into meaningful units
def split_translation(text):
    # Remove bracketed comments
    text = re.sub(r'\[.*?\]', '', text)
    # Split by space or punctuation
    units = [unit.strip() for unit in re.split(r'[,\s]+', text) if unit.strip()]
    return units

# Function to check if a string is numeric (including Sinhala numerals if needed)
def is_number(s):
    return s.isdigit()

# List to store word-meaning pairs
word_meaning_pairs = []

# Read the input CSV
with open(input_file, 'r', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    print("CSV Headers:", reader.fieldnames)

    for idx, row in enumerate(reader, start=1):
        transliteration = row.get('අක්ෂර පරිවර්තනය', '')
        translation = row.get('පරිවර්තනය', '')
        inscription_id = row.get('සෙල්ලිපි අංකය', f"ID_{idx}")

        if not transliteration or not translation:
            continue

        trans_words = split_transliteration(transliteration)
        trans_units = split_translation(translation)

        min_length = min(len(trans_words), len(trans_units))
        if len(trans_words) != len(trans_units):
            print(f"Warning: Mismatch in row {idx}: {len(trans_words)} transliteration words, {len(trans_units)} translation units")

        # Add blank row + inscription ID
        word_meaning_pairs.append({
            'word': '',
            'meaning': '',
            'සෙල්ලිපි අංකය': inscription_id
        })

        for i in range(min_length):
            word = trans_words[i]
            meaning = trans_units[i]
            if is_number(word) or is_number(meaning):
                continue  # Skip numeric-only words or meanings

            word_meaning_pairs.append({
                'word': word,
                'meaning': meaning,
                'සෙල්ලිපි අංකය': inscription_id
            })

# Write output with BOM so Excel reads Sinhala properly
with open(output_file, 'w', encoding='utf-8-sig', newline='') as csvfile:
    fieldnames = ['word', 'meaning', 'සෙල්ලිපි අංකය']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for pair in word_meaning_pairs:
        writer.writerow(pair)

print(f"Output written to {output_file}")
