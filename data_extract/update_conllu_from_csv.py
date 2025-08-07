# import pandas as pd
# import re

# # Load the filtered CSV file
# csv_path = "filtered_inscriptions.csv"
# filtered_df = pd.read_csv(csv_path)

# # Initialize list to hold rows for the new DataFrame
# conllu_rows = []

# # Process each sentence in the CSV file
# for _, row in filtered_df.iterrows():
#     sentence = str(row['පරිවර්තනය']).strip()
#     if sentence:
#         words = sentence.split()
#         for idx, word in enumerate(words, start=1):
#             # Remove brackets and periods
#             cleaned_word = re.sub(r"[.\[\]]", "", word)
#             conllu_rows.append({
#                 "ID": idx,
#                 "FORM": cleaned_word,
#                 "LEMMA": cleaned_word,
#                 "UPOS": "",
#                 "XPOS": "",
#                 "FEATS": "",
#                 "HEAD": "",
#                 "DEPREL": "",
#                 "DEPS": "",
#                 "MISC": ""
#             })
#         # Add a blank row to separate sentences
#         conllu_rows.append({col: "" for col in ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]})

# # Create and export the DataFrame
# new_conllu_df = pd.DataFrame(conllu_rows)
# new_conllu_df.to_excel("updated_conllu_data.xlsx", index=False)

import pandas as pd
import re

# Load the filtered CSV file
csv_path = "filtered_inscriptions.csv"
filtered_df = pd.read_csv(csv_path)

# Initialize list to hold rows for the new DataFrame
conllu_rows = []

def clean_word(word):
    # Remove stray punctuation
    word = re.sub(r"[.\[\]()]", "", word)

    # Specific replacements
    if word == "ලෙ​ණේ":
        return "ලෙ​ණ"
    if word == "ලෙනේ":
        return "ලෙන"

    # Remove suffixes
    if word.endswith("ගේ"):
        word = word[:-2]  # remove only 'ගේ' (2 letters: 'ග', 'ේ')
    elif word.endswith("ට"):
        word = word[:-1]

    return word

# Process each sentence
for _, row in filtered_df.iterrows():
    sentence = str(row['පරිවර්තනය']).strip()

    if sentence:
        # Remove anything inside [] or ()
        sentence_cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", sentence)

        words = sentence_cleaned.split()

        for idx, word in enumerate(words, start=1):
            cleaned_word = clean_word(word)
            if cleaned_word:  # avoid adding empty strings
                conllu_rows.append({
                    "ID": idx,
                    "FORM": cleaned_word,
                    "LEMMA": cleaned_word,
                    "UPOS": "",
                    "XPOS": "",
                    "FEATS": "",
                    "HEAD": "",
                    "DEPREL": "",
                    "DEPS": "",
                    "MISC": ""
                })

        # Separate sentences
        conllu_rows.append({col: "" for col in ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]})

# Export
new_conllu_df = pd.DataFrame(conllu_rows)
new_conllu_df.to_excel("updated_conllu_data.xlsx", index=False)

print("updated_conllu_data.xlsx created.")


