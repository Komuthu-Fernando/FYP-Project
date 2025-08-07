# import pandas as pd
# import re

# # Load CSV file (change file name/path if needed)
# file_path = "filtered_inscriptions 1.csv"
# df = pd.read_csv(file_path)

# # Choose the column with Sinhala text
# text_column = "අක්ෂර පරිවර්තනය"

# # Function to clean and split Sinhala words
# def extract_words(text):
#     text = re.sub(r"[0-9()\[\]]", "", str(text))  # Remove numbers and brackets
#     text = text.replace("-", " ")                # Replace hyphens with spaces
#     words = text.split()                         # Split into words
#     return words

# # Apply to all rows in the selected column and flatten the result
# all_words = df[text_column].dropna().apply(extract_words).sum()

# # Get unique words, sort alphabetically
# unique_sorted_words = sorted(set(all_words))

# # Save to a CSV file (one word per row)
# output_csv = "sorted_sinhala_words.csv"
# pd.DataFrame(unique_sorted_words, columns=["Word"]).to_csv(output_csv, index=False, encoding="utf-8-sig")

# print(f"Saved {len(unique_sorted_words)} words to {output_csv}")

import pandas as pd

# Load sorted Sinhala words CSV
sorted_words_df = pd.read_csv("sorted_sinhala_words.csv")
sorted_words = set(sorted_words_df["Word"].dropna().astype(str).str.strip())

# Load comparison Excel file
comparison_df = pd.read_excel("Copy of word_comparison_output (2).xlsx")

# DEBUG: Show column names
print("Excel columns:", comparison_df.columns.tolist())

# Use the correct column name from the print above
comparison_words = set(comparison_df["Word"].dropna().astype(str).str.strip())

# Find missing words
missing_words = sorted_words - comparison_words

# Save to CSV
missing_df = pd.DataFrame(sorted(missing_words), columns=["Missing Words"])
missing_df.to_csv("missing_words_from_comparison.csv", index=False, encoding="utf-8-sig")

print(f"Saved {len(missing_words)} missing words to missing_words_from_comparison.csv")
